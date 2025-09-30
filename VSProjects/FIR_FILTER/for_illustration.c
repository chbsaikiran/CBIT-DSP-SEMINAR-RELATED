#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define FRAME_SIZE 4
#define FILT_SIZE 2

float delay_line[(FILT_SIZE - 1) + FRAME_SIZE];

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

int main(void)
{
    FILE *fcoeffs, *finput, *fout;
    float in[FRAME_SIZE], coeffs[FILT_SIZE],out[FRAME_SIZE];
    int i,j;

    fcoeffs = fopen("..\\..\\PythonProjects\\FIR_FILTER\\filter_coeffs.bin","rb");
    finput = fopen("..\\..\\PythonProjects\\FIR_FILTER\\test_signal.bin", "rb");
    fout = fopen("out_msvc_wo_circ_buffer.bin","wb");

    fread(coeffs,FILT_SIZE,sizeof(float),fcoeffs);
    
    while(1)
    {
        temp = fread(in, sizeof(float), FRAME_SIZE, finput);
        if (temp < FRAME_SIZE)
            break;

        fir_filter(in, coeffs, out, FILT_SIZE, FRAME_SIZE);

        fwrite(out,FRAME_SIZE,sizeof(float),fout);
    }

    fclose(fcoeffs);
    fclose(finput);
    fclose(fout);

    return 0;
}