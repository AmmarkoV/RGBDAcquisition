/*
*levmar.c, levmar.h provided under the MIT license.
*
*From : https://gist.github.com/rbabich/3539146
*Copyright (c) 2008-2016 Ron Babich
*
*Permission is hereby granted, free of charge, to any person
*obtaining a copy of this software and associated documentation
*files (the "Software"), to deal in the Software without
*restriction, including without limitation the rights to use,
*copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the
*Software is furnished to do so, subject to the following
*conditions:
*
*The above copyright notice and this permission notice shall be
*included in all copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
vEXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
*OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
*NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
*HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
*WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
*OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef LEVMAR_H_INCLUDED
#define LEVMAR_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif


typedef struct
{
    int verbose;
    int max_it;
    double init_lambda;
    double up_factor;
    double down_factor;
    double target_derr;
    int final_it;
    double final_err;
    double final_derr;
} LMstat;

void levmarq_init(LMstat *lmstat);

int levmarq(
    int npar,
    double *par,
    int ny,
    double *y,
    double *dysq,
    double (*func) (double *, int, void *),
    void (*grad)(double *, double *, int, void *),
    void *fdata,
    LMstat *lmstat
);

double error_func(
    double *par,
    int ny,
    double *y,
    double *dysq,
    double (*func)(double *, int, void *),
    void *fdata
);

void solve_axb_cholesky(int n, double l[n][n], double x[n], double b[n]);

int cholesky_decomp(int n, double l[n][n], double a[n][n]);

#ifdef __cplusplus
}
#endif

#endif
