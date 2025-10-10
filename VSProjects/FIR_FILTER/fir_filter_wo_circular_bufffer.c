#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define USE_FIXED_PT_CODE
#define USE_ARM_NEON_OPT
#define PROFILE_CODE

#ifdef USE_ARM_NEON_OPT
#include "arm_neon.h"
#endif

#ifdef PROFILE_CODE
#include <sys/time.h>
#endif

#define ABS_FLOAT(x) ((x) > 0 ? (x):(0-(x)))

#define Word64 long long
#define Word32 int
#define Word16 short

#define FRAME_SIZE 30
#define FILT_SIZE 13
#define COEFF_TYPE Word32
#define INPT_TYPE Word32
#define INTER_TYPE Word64
#define COEFF_PRECISION_BITS 31
#define INPT_PRECISION_BITS 31
#define INTER_PRECISION_BITS 31

INTER_TYPE s64_mul_s32_s32(COEFF_TYPE x, INPT_TYPE y)
{
    INTER_TYPE prod;

    prod = ((INTER_TYPE)x)*((INTER_TYPE)y);

    return prod;
}

INTER_TYPE s64_mla_s32_s32(INTER_TYPE sum, COEFF_TYPE x, INPT_TYPE y)
{
    INTER_TYPE prod;

    prod = ((INTER_TYPE)x)*((INTER_TYPE)y);

    sum = sum + prod;

    return sum;
}

Word32 float_to_fixed_conv(float x, Word32 qfactor)
{
    return ((Word32)(x*(pow(2,qfactor))));
}

Word16 float_to_fixed_conv_16bit(float x, Word16 qfactor)
{
    return ((Word16)(x*(pow(2,qfactor))));
}

float fixed_to_float_conv(Word32 x, Word32 qfactor)
{
    return (((float)x)/((float)(pow(2,qfactor))));
}

float fixed_to_float_conv_16bit(Word16 x, Word16 qfactor)
{
    return (((float)x) / ((float)(pow(2, qfactor))));
}

#ifdef USE_FIXED_PT_CODE
INPT_TYPE delay_line_fxd[(FILT_SIZE - 1) + FRAME_SIZE];
#else
float delay_line[(FILT_SIZE - 1) + FRAME_SIZE];
#endif

#ifdef USE_FIXED_PT_CODE
#ifndef USE_ARM_NEON_OPT
void fir_filter_fxd_pt(INPT_TYPE* in, COEFF_TYPE* coeffs, INPT_TYPE* out, INPT_TYPE num_of_filt_coeffs, INPT_TYPE frame_size)
{
    INTER_TYPE acc;     // accumulator for MACs
    COEFF_TYPE*coeffp; // pointer to coefficients
    INPT_TYPE*inputp; // pointer to input samples
    INPT_TYPE n;
    INPT_TYPE k;
 
    // put the new samples at the high end of the buffer
    memcpy( &delay_line_fxd[num_of_filt_coeffs - 1], in,
            frame_size * sizeof(INPT_TYPE) );
 
    // apply the filter to each input sample
    for ( n = 0; n < frame_size; n++ ) 
    {
        // calculate out n
        coeffp = coeffs;
        inputp = &delay_line_fxd[num_of_filt_coeffs - 1 + n];
        acc = 0;
        for ( k = 0; k < num_of_filt_coeffs; k++ ) 
        {
            acc = s64_mla_s32_s32(acc, (*coeffp++), (*inputp--));
        }
        //acc = acc << (64 - 32 - 3);
        //out[n] = (INPT_TYPE)(((acc >> 46) + 1) >> 1);
        out[n] = (INPT_TYPE)(acc >> (INTER_PRECISION_BITS));
    }
    // shift input samples back in time for next time
    memmove( &delay_line_fxd[0], &delay_line_fxd[frame_size],
            (num_of_filt_coeffs - 1) * sizeof(INPT_TYPE));
 
}
#else
void fir_filter_fxd_pt(INPT_TYPE* in, COEFF_TYPE* coeffs, INPT_TYPE* out,INPT_TYPE num_of_filt_coeffs, INPT_TYPE frame_size)
{
    Word64 acc1,acc2;     // accumulator for MACs
    COEFF_TYPE *coeffp; // pointer to coefficients
    INPT_TYPE *inputp,*inputp1; // pointer to input samples
    INPT_TYPE n,len,rem;
    INPT_TYPE k;
    int32x4_t q0,q1,q2,q3;
    int64x2_t q5,q4,q6,q7;
    int64x1_t d12,d13,d14,d15;
    int32x2_t d0;
 
    // put the new samples at the high end of the buffer
    memcpy( &delay_line_fxd[num_of_filt_coeffs - 1], in,
            frame_size * sizeof(int) );
 
    // apply the filter to each input sample
    for ( n = 0; n < frame_size; n+=2 ) 
    {
        // calculate out n
        coeffp = &coeffs[num_of_filt_coeffs-1];
        inputp = &delay_line_fxd[num_of_filt_coeffs + n];
        inputp1 = inputp - 1;
        acc1 = 0;
        acc2 = 0;
        len = num_of_filt_coeffs >> 2;
        rem = num_of_filt_coeffs & 3;
        while(rem--)
        {
            acc1 = s64_mla_s32_s32(acc1, (*coeffp), (*inputp--));
            acc2 = s64_mla_s32_s32(acc2, (*coeffp), (*inputp1--));
            coeffp--;
        }
        d14 = vdup_n_s64(acc2);
        d15 = vdup_n_s64(acc1);
        q7 = vcombine_s64(d14,d15);
        coeffp = coeffp - 3;
        inputp = inputp - 3;
        inputp1 = inputp1 - 3;
        q4 = vdupq_n_s64(0);
        q5 = vdupq_n_s64(0);
        while(len--) 
        {
            //acc = s64_mla_s32_s32(acc, (*coeffp--), (*inputp--));
            q0 = vld1q_s32(((int32_t*)coeffp)); //3 2 1 0
            q1 = vld1q_s32(((int32_t*)inputp)); //7 6 5 4
            inputp -= 4;
            coeffp -= 4;
            q2 = vld1q_s32(((int32_t*)inputp1)); //6 5 4 3
            inputp1 -= 4;
            q5 = vmlal_s32(q5,vget_low_s32(q0),vget_low_s32(q1));
            q4 = vmlal_s32(q4,vget_low_s32(q0),vget_low_s32(q2));
            q5 = vmlal_s32(q5,vget_high_s32(q0),vget_high_s32(q1));
            q4 = vmlal_s32(q4,vget_high_s32(q0),vget_high_s32(q2));
        }
        d12 = vadd_s64(vget_low_s64(q4),vget_high_s64(q4));
        d13 = vadd_s64(vget_low_s64(q5),vget_high_s64(q5));
        q6 = vcombine_s64(d12,d13);
        q6 = vaddq_s64(q7,q6);
        d0 = vshrn_n_s64(q6,31);
        //out[n] = (Word32)(acc >> 31);
        vst1_s32(((int32_t*)&out[n]), d0);
    }
    // shift input samples back in time for next time
    memcpy( &delay_line_fxd[0], &delay_line_fxd[frame_size],
            (num_of_filt_coeffs - 1) * sizeof(int));
 
}
#endif
#else
void fir_filter(float* in, float* coeffs, float* out, Word32 num_of_filt_coeffs, INPT_TYPE frame_size)
{
    float acc;     // accumulator for MACs
    float* coeffp; // pointer to coefficients
    float* inputp; // pointer to input samples
    INPT_TYPE n;
    INPT_TYPE k;

    // put the new samples at the high end of the buffer
    memcpy(&delay_line[num_of_filt_coeffs - 1], in,
        frame_size * sizeof(float));

    // apply the filter to each input sample
    for (n = 0; n < frame_size; n++)
    {
        // calculate out n
        coeffp = coeffs;
        inputp = &delay_line[num_of_filt_coeffs - 1 + n];
        acc = 0;
        for (k = 0; k < num_of_filt_coeffs; k++)
        {
            acc += (*coeffp++)*(*inputp--);
        }
        out[n] = acc;
    }
    // shift input samples back in time for next time
    memmove(&delay_line[0], &delay_line[frame_size],
        (num_of_filt_coeffs - 1) * sizeof(float));

}
#endif

int main(void)
{
    FILE *fcoeffs, *finput, *fout;
    float in[FRAME_SIZE], coeffs[FILT_SIZE],out[FRAME_SIZE];
    COEFF_TYPE coeffs_fxd_pt[FILT_SIZE];
    INPT_TYPE in_fxd_pt[FRAME_SIZE], temp, out_fxd_pt[FRAME_SIZE];
    int i,j;
#ifdef PROFILE_CODE
    long seconds;
    long microseconds;
    double elapsed = 0;
#endif

    fcoeffs = fopen("filter_coeffs.bin","rb");
    finput = fopen("test_signal.bin", "rb");
#ifdef USE_FIXED_PT_CODE
#ifdef USE_ARM_NEON_OPT
    fout = fopen("out_linux_wo_circ_buffer_arm_opt.bin","wb");
#else
    fout = fopen("out_linux_wo_circ_buffer_wo_arm_opt.bin","wb");
#endif
#else
    fout = fopen("out_linux_wo_circ_buffer_float.bin","wb");
#endif

    fread(coeffs,FILT_SIZE,sizeof(float),fcoeffs);
#ifdef USE_FIXED_PT_CODE
    if (sizeof(COEFF_TYPE) == 4)
    {
        for (i = 0; i < FILT_SIZE; i++)
        {
            coeffs_fxd_pt[i] = float_to_fixed_conv(coeffs[i], (COEFF_PRECISION_BITS - 5)); //for using Gaurd bits 2, without Gaurd bits 5
        }
    }
    else
    {
        for (i = 0; i < FILT_SIZE; i++)
        {
            coeffs_fxd_pt[i] = float_to_fixed_conv_16bit(coeffs[i], (COEFF_PRECISION_BITS - 5)); //for using Gaurd bits 2, without Gaurd bits 5
        }
    }

    for (i = 0; i < ((FILT_SIZE - 1)+FRAME_SIZE); i++)
    {
        delay_line_fxd[i] = 0;
    }

#ifdef USE_ARM_NEON_OPT
    for (i = 0; i < FILT_SIZE/2; i++)
    {
        temp = coeffs_fxd_pt[i];
        coeffs_fxd_pt[i] = coeffs_fxd_pt[FILT_SIZE - i - 1];
        coeffs_fxd_pt[FILT_SIZE - i - 1] = temp;
    }
#endif
#endif
    
    while(1)
    {
        temp = fread(in, sizeof(float), FRAME_SIZE, finput);
        if (temp < FRAME_SIZE)
            break;
#ifdef USE_FIXED_PT_CODE
        if (sizeof(INPT_TYPE) == 4)
        {
            for (i = 0; i < FRAME_SIZE; i++)
            {
                in_fxd_pt[i] = float_to_fixed_conv(in[i], (INPT_PRECISION_BITS - 3));
            }
        }
        else
        {
            for (i = 0; i < FRAME_SIZE; i++)
            {
                in_fxd_pt[i] = float_to_fixed_conv_16bit(in[i], (INPT_PRECISION_BITS - 3));
            }
        }
#ifdef PROFILE_CODE
        struct timeval start, end;
        gettimeofday(&start, NULL);
#endif
        fir_filter_fxd_pt(in_fxd_pt, coeffs_fxd_pt, out_fxd_pt, FILT_SIZE, FRAME_SIZE);
#ifdef PROFILE_CODE
        gettimeofday(&end, NULL);
        seconds = (end.tv_sec - start.tv_sec);
        microseconds = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
        elapsed += microseconds*1e-6;
#endif
        if (sizeof(INPT_TYPE) == 4)
        {
            for (i = 0; i < FRAME_SIZE; i++)
            {
                out[i] = fixed_to_float_conv(out_fxd_pt[i], (INPT_PRECISION_BITS - 8));
            }
        }
        else
        {
            for (i = 0; i < FRAME_SIZE; i++)
            {
                out[i] = fixed_to_float_conv_16bit(out_fxd_pt[i], (INPT_PRECISION_BITS - 8));
            }
        }
#else
        fir_filter(in, coeffs, out, FILT_SIZE, FRAME_SIZE);
#endif
        fwrite(out,FRAME_SIZE,sizeof(float),fout);
    }
#ifdef PROFILE_CODE
    printf("elapsed_time = %lf\n",elapsed);
#endif

    fclose(fcoeffs);
    fclose(finput);
    fclose(fout);

    return 0;
}
