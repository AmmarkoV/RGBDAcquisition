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


/**
* @brief LMstat Levenberg Marquardt solver context
* @ingroup Levmar
*/
struct LMstat
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
} ;


/**
* @brief Solve the equation Ax=b for a symmetric positive-definite matrix A,
*        using the Cholesky decomposition A=LL^T.  The matrix L is passed in "l".
*        Elements above the diagonal are ignored.
* @ingroup Levmar
* @param  n matrix dimension
* @param  lower diagonal matrix l[n][n]
* @param  symmetric positive-definite matrix a[n][n]
* @return 0=Success/1=Failure
*/
void levmar_solveAXBUsingCholesky(int n, double l[n][n], double x[n], double b[n]);


/**
* @brief This function takes a symmetric, positive-definite matrix "a" and returns
*         its (lower-triangular) Cholesky factor in "l".  Elements above the
*         diagonal are neither used nor modified.  The same array may be passed
*         as both l and a, in which case the decomposition is performed in place.
* @ingroup Levmar
* @param  n matrix dimension
* @param  lower diagonal matrix l[n][n]
* @param  symmetric positive-definite matrix a[n][n]
* @return 0=Success/1=Failure
*/
int levmar_choleskyDecomposition(int n, double l[n][n], double a[n][n]);

/**
* @brief Initialize a Levenberg Marquardt solver
* @ingroup Levmar
* @param  LMstat Structure that will hold the solving session
* @return 1=Success/0=Failure
*/
int levmar_initialize(struct LMstat *lmstat);



/**
* @brief Perform least-squares minimization using the Levenberg-Marquardt algorithm.
* @ingroup Levmar
* @param npar number of parameters
* @param par array of parameters to be varied
* @param ny number of measurements to be fit
* @param y array of measurements
* @param dysq array of error in measurements, squared (set dysq=NULL for unweighted least-squares)
* @param func function to be fit
* @param grad gradient of "func" with respect to the input parameters
* @param fdata pointer to any additional data required by the function
* @param lmstat pointer to the "status" structure, where minimization parameters are set and the final status is returned.
* @return 1=Success/0=Failure
*/
int levmar_solve(
                 int npar,
                 double *par,
                 int ny,
                 double *y,
                 double *dysq,
                 double (*func) (double *, int, void *),
                 void (*grad)(double *, double *, int, void *),
                 void *fdata,
                 struct LMstat *lmstat
                );


/**
* @brief Calculate the error function (chi-squared)
* @ingroup Levmar
* @param par array of parameters to be varied
* @param ny number of measurements to be fit
* @param y array of measurements
* @param dysq array of error in measurements, squared (set dysq=NULL for unweighted least-squares)
* @param func function to be fit
* @param fdata pointer to any additional data required by the function
* @return Error function
*/
double levmar_errorFunction(
                            double *par,
                            int ny,
                            double *y,
                            double *dysq,
                            double (*func)(double *, int, void *),
                            void *fdata
                           );



#ifdef __cplusplus
}
#endif

#endif
