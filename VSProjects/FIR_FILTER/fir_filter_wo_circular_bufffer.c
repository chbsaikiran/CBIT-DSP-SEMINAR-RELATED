#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define USE_FIXED_PT_CODE
//#define PROFILE_CODE

#ifdef PROFILE_CODE
#include <sys/time.h>
#endif

#define ABS_FLOAT(x) ((x) > 0 ? (x):(0-(x)))

#define Word64 long long
#define Word32 int
#define Word16 short

#define FRAME_SIZE 30
#define FILT_SIZE 13
#define COEFF_TYPE Word16
#define INPT_TYPE Word16
#define INTER_TYPE Word64
#define COEFF_PRECISION_BITS 15
#define INPT_PRECISION_BITS 15
#define INTER_PRECISION_BITS 15

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
        out[n] = (INPT_TYPE)(((acc >> (INTER_PRECISION_BITS- 1)) + 1) >> 1);
    }
    // shift input samples back in time for next time
    memmove( &delay_line_fxd[0], &delay_line_fxd[frame_size],
            (num_of_filt_coeffs - 1) * sizeof(INPT_TYPE));
 
}
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

    fcoeffs = fopen("..\\..\\PythonProjects\\FIR_FILTER\\filter_coeffs.bin","rb");
    finput = fopen("..\\..\\PythonProjects\\FIR_FILTER\\test_signal.bin", "rb");
    fout = fopen("out_msvc_wo_circ_buffer.bin","wb");

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
