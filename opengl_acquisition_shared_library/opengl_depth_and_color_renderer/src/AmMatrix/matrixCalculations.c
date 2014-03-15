

#include "matrixCalculations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix3x3Tools.h"
#include "matrix4x4Tools.h"
#include "solveLinearSystemGJ.h"
#include "solvePnPIterative.h"



#define Sgn(x)              ( (x)<0 ? -1:1 )    /* Sgn(0) = 1 ! */
#define REAL_ZERO(x)        ( (x<0.0001) && (x>0.0001) ? 0:1 )    /* REAL_ZERO(0.000001) = 1 ! */
#define MAX(a,b)            ( (a)<(b) ? b:a )

int
icvSingularValueDecomposition( int M,
                               int N,
                               double *A,
                               double *W, int get_U, double *U, int get_V, double *V )
{
    int i = 0, j, k, l = 0, i1, k1, l1 = 0;
    int iterations, error = 0, jN, iN, kN, lN = 0;
    double *rv1;
    double c, f, g, h, s, x, y, z, scale, anorm;
    double af, ag, ah, t;
    int MN = M * N;
    int NN = N * N;

    /*  max_iterations - maximum number QR-iterations
       cc - reduces requirements to number stitch (cc>1)
     */

    int max_iterations = 100;
    double cc = 100;

    if( M < N )
        return N;

    rv1 = (double *) malloc( N * sizeof( double ));

    if( rv1 == 0 )
        return N;

    for( iN = 0; iN < MN; iN += N )
    {
        for( j = 0; j < N; j++ )
            U[iN + j] = A[iN + j];
    }                           /* for */

    /*  Adduction to bidiagonal type (transformations of reflection).
       Bidiagonal matrix is located in W (diagonal elements)
       and in rv1 (upperdiagonal elements)
     */

    g = 0;
    scale = 0;
    anorm = 0;

    for( i = 0, iN = 0; i < N; i++, iN += N )
    {

        l = i + 1;
        lN = iN + N;
        rv1[i] = scale * g;

        /*  Multiplyings on the left  */

        g = 0;
        s = 0;
        scale = 0;

        for( kN = iN; kN < MN; kN += N )
            scale += fabs( U[kN + i] );

        if( !REAL_ZERO( scale ))
        {

            for( kN = iN; kN < MN; kN += N )
            {

                U[kN + i] /= scale;
                s += U[kN + i] * U[kN + i];
            }                   /* for */

            f = U[iN + i];
            g = -sqrt( s ) * Sgn( f );
            h = f * g - s;
            U[iN + i] = f - g;

            for( j = l; j < N; j++ )
            {

                s = 0;

                for( kN = iN; kN < MN; kN += N )
                {

                    s += U[kN + i] * U[kN + j];
                }               /* for */

                f = s / h;

                for( kN = iN; kN < MN; kN += N )
                {

                    U[kN + j] += f * U[kN + i];
                }               /* for */
            }                   /* for */

            for( kN = iN; kN < MN; kN += N )
                U[kN + i] *= scale;
        }                       /* if */

        W[i] = scale * g;

        /*  Multiplyings on the right  */

        g = 0;
        s = 0;
        scale = 0;

        for( k = l; k < N; k++ )
            scale += fabs( U[iN + k] );

        if( !REAL_ZERO( scale ))
        {

            for( k = l; k < N; k++ )
            {

                U[iN + k] /= scale;
                s += (U[iN + k]) * (U[iN + k]);
            }                   /* for */

            f = U[iN + l];
            g = -sqrt( s ) * Sgn( f );
            h = f * g - s;
            U[i * N + l] = f - g;

            for( k = l; k < N; k++ )
                rv1[k] = U[iN + k] / h;

            for( jN = lN; jN < MN; jN += N )
            {

                s = 0;

                for( k = l; k < N; k++ )
                    s += U[jN + k] * U[iN + k];

                for( k = l; k < N; k++ )
                    U[jN + k] += s * rv1[k];

            }                   /* for */

            for( k = l; k < N; k++ )
                U[iN + k] *= scale;
        }                       /* if */

        t = fabs( W[i] );
        t += fabs( rv1[i] );
        anorm = MAX( anorm, t );
    }                           /* for */

    anorm *= cc;

    /*  accumulation of right transformations, if needed  */

    if( get_V )
    {

        for( i = N - 1, iN = NN - N; i >= 0; i--, iN -= N )
        {

            if( i < N - 1 )
            {

                /*  pass-by small g  */
                if( !REAL_ZERO( g ))
                {

                    for( j = l, jN = lN; j < N; j++, jN += N )
                        V[jN + i] = U[iN + j] / U[iN + l] / g;

                    for( j = l; j < N; j++ )
                    {

                        s = 0;

                        for( k = l, kN = lN; k < N; k++, kN += N )
                            s += U[iN + k] * V[kN + j];

                        for( kN = lN; kN < NN; kN += N )
                            V[kN + j] += s * V[kN + i];
                    }           /* for */
                }               /* if */

                for( j = l, jN = lN; j < N; j++, jN += N )
                {
                    V[iN + j] = 0;
                    V[jN + i] = 0;
                }               /* for */
            }                   /* if */

            V[iN + i] = 1;
            g = rv1[i];
            l = i;
            lN = iN;
        }                       /* for */
    }                           /* if */

    /*  accumulation of left transformations, if needed  */

    if( get_U )
    {

        for( i = N - 1, iN = NN - N; i >= 0; i--, iN -= N )
        {

            l = i + 1;
            lN = iN + N;
            g = W[i];

            for( j = l; j < N; j++ )
                U[iN + j] = 0;

            /*  pass-by small g  */
            if( !REAL_ZERO( g ))
            {

                for( j = l; j < N; j++ )
                {

                    s = 0;

                    for( kN = lN; kN < MN; kN += N )
                        s += U[kN + i] * U[kN + j];

                    f = s / U[iN + i] / g;

                    for( kN = iN; kN < MN; kN += N )
                        U[kN + j] += f * U[kN + i];
                }               /* for */

                for( jN = iN; jN < MN; jN += N )
                    U[jN + i] /= g;
            }
            else
            {

                for( jN = iN; jN < MN; jN += N )
                    U[jN + i] = 0;
            }                   /* if */

            U[iN + i] += 1;
        }                       /* for */
    }                           /* if */

    /*  Iterations QR-algorithm for bidiagonal matrixes
       W[i] - is the main diagonal
       rv1[i] - is the top diagonal, rv1[0]=0.
     */

    for( k = N - 1; k >= 0; k-- )
    {

        k1 = k - 1;
        iterations = 0;

        for( ;; )
        {

            /*  Cycle: checking a possibility of fission matrix  */
            for( l = k; l >= 0; l-- )
            {

                l1 = l - 1;

                if( REAL_ZERO( rv1[l] ) || REAL_ZERO( W[l1] ))
                    break;
            }                   /* for */

            if( !REAL_ZERO( rv1[l] ))
            {

                /*  W[l1] = 0,  matrix possible to fission
                   by clearing out rv1[l]  */

                c = 0;
                s = 1;

                for( i = l; i <= k; i++ )
                {

                    f = s * rv1[i];
                    rv1[i] = c * rv1[i];

                    /*  Rotations are done before the end of the block,
                       or when element in the line is finagle.
                     */

                    if( REAL_ZERO( f ))
                        break;

                    g = W[i];

                    /*  Scaling prevents finagling H ( F!=0!) */

                    af = fabs( f );
                    ag = fabs( g );

                    if( af < ag )
                        h = ag * sqrt( 1 + (f / g) * (f / g) );
                    else
                        h = af * sqrt( 1 + (f / g) * (f / g) );

                    W[i] = h;
                    c = g / h;
                    s = -f / h;

                    if( get_U )
                    {

                        for( jN = 0; jN < MN; jN += N )
                        {

                            y = U[jN + l1];
                            z = U[jN + i];
                            U[jN + l1] = y * c + z * s;
                            U[jN + i] = -y * s + z * c;
                        }       /* for */
                    }           /* if */
                }               /* for */
            }                   /* if */


            /*  Output in this place of program means,
               that rv1[L] = 0, matrix fissioned
               Iterations of the process of the persecution
               will be executed always for
               the bottom block ( from l before k ),
               with increase l possible.
             */

            z = W[k];

            if( l == k )
                break;

            /*  Completion iterations: lower block
               became trivial ( rv1[K]=0)  */

            if( iterations++ == max_iterations )
                return k;

            /*  Shift is computed on the lowest order 2 minor.  */

            x = W[l];
            y = W[k1];
            g = rv1[k1];
            h = rv1[k];

            /*  consequent fission prevents forming a machine zero  */
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h) / y;

            /*  prevented overflow  */
            if( fabs( f ) > 1 )
            {
                g = fabs( f );
                g *= sqrt( 1 + (1 / f) * (1 / f) );
            }
            else
                g = sqrt( f * f + 1 );

            f = ((x - z) * (x + z) + h * (y / (f + fabs( g ) * Sgn( f )) - h)) / x;
            c = 1;
            s = 1;

            for( i1 = l; i1 <= k1; i1++ )
            {

                i = i1 + 1;
                g = rv1[i];
                y = W[i];
                h = s * g;
                g *= c;

                /*  Scaling at calculation Z prevents its clearing,
                   however if F and H both are zero - pass-by of fission on Z.
                 */

                af = fabs( f );
                ah = fabs( h );

                if( af < ah )
                    z = ah * sqrt( 1 + (f / h) * (f / h) );

                else
                {

                    z = 0;
                    if( !REAL_ZERO( af ))
                        z = af * sqrt( 1 + (h / f) * (h / f) );
                }               /* if */

                rv1[i1] = z;

                /*  if Z=0, the rotation is free.  */
                if( !REAL_ZERO( z ))
                {

                    c = f / z;
                    s = h / z;
                }               /* if */

                f = x * c + g * s;
                g = -x * s + g * c;
                h = y * s;
                y *= c;

                if( get_V )
                {

                    for( jN = 0; jN < NN; jN += N )
                    {

                        x = V[jN + i1];
                        z = V[jN + i];
                        V[jN + i1] = x * c + z * s;
                        V[jN + i] = -x * s + z * c;
                    }           /* for */
                }               /* if */

                af = fabs( f );
                ah = fabs( h );

                if( af < ah )
                    z = ah * sqrt( 1 + (f / h) * (f / h) );
                else
                {

                    z = 0;
                    if( !REAL_ZERO( af ))
                        z = af * sqrt( 1 + (h / f) * (h / f) );
                }               /* if */

                W[i1] = z;

                if( !REAL_ZERO( z ))
                {

                    c = f / z;
                    s = h / z;
                }               /* if */

                f = c * g + s * y;
                x = -s * g + c * y;

                if( get_U )
                {

                    for( jN = 0; jN < MN; jN += N )
                    {

                        y = U[jN + i1];
                        z = U[jN + i];
                        U[jN + i1] = y * c + z * s;
                        U[jN + i] = -y * s + z * c;
                    }           /* for */
                }               /* if */
            }                   /* for */

            rv1[l] = 0;
            rv1[k] = f;
            W[k] = x;
        }                       /* for */

        if( z < 0 )
        {

            W[k] = -z;

            if( get_V )
            {

                for( jN = 0; jN < NN; jN += N )
                    V[jN + k] *= -1;
            }                   /* if */
        }                       /* if */
    }                           /* for */

    free( rv1 );

    return error;

}                               /* vm_SingularValueDecomposition */



int convertRodriguezTo3x3(double * result,double * matrix)
{
  if ( (matrix==0) ||  (result==0) ) { return 0; }


  double x = matrix[0] , y = matrix[1] , z = matrix[2];
  double th = sqrt( x*x + y*y + z*z );
  double cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  if ( th < 0.00001 )
    {
       create3x3IdentityMatrix(result);
       return 1;
    }

   //NORMAL RESULT
   result[0]=x*x * (1 - cosTh) + cosTh;          result[1]=x*y*(1 - cosTh) - z*sin(th);      result[2]=x*z*(1 - cosTh) + y*sin(th);
   result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;          result[5]=y*z*(1 - cosTh) - x*sin(th);
   result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;

  fprintf(stderr,"rodriguez %f %f %f\n ",matrix[0],matrix[1],matrix[2]);
  print3x3DMatrix("Rodriguez Initial", result);

  return 1;
}


void changeYandZAxisOpenGL4x4Matrix(double * result,double * matrix)
{
  fprintf(stderr,"Invert Y and Z axis\n");
  double * invertOp = (double * ) malloc ( sizeof(double) * 16 );
  if (invertOp==0) { return; }

  create4x4IdentityMatrix(invertOp);
  invertOp[5]=-1;   invertOp[10]=-1;
  multiplyTwo4x4Matrices(result, matrix, invertOp);
  free(invertOp);
}

int projectPointsFrom3Dto2D(double * x2D, double * y2D , double * x3D, double *y3D , double * z3D , double * intrinsics , double * rotation3x3 , double * translation)
{
  double fx = intrinsics[0];
  double fy = intrinsics[4];
  double cx = intrinsics[2];
  double cy = intrinsics[5];

  double * t = translation;
  double * r = rotation3x3;

  //Result
  //fx * t0 + cx * t2 + (x3D) * ( fx * r0 + cx * r6 )  + (y3D) * ( fx * r1 + cx * r7 ) + (z3D) * (fx * r2 +cx * r8) / t3 + r7 x3D + r8 * y3D + r9 * z3D
  //fy * t1 + cy * t2 + x3D * ( fy * r3 + cy * r6 )  + y3D * ( fy * r4 + cy * r7 ) + z3D * (fy * r5 +cy * r8) / t3 + r7 x3D + r8 * y3D + r9 * z3D
  //1

  double x2DBuf =  fx * t[0] + cx * t[2] + (*x3D) * ( fx * r[0] + cx * r[6] )  + (*y3D) * ( fx * r[1] + cx * r[7] ) + (*z3D) * (fx * r[2] +cx * r[8]);
  double y2DBuf =  fy * t[1] + cy * t[2] + (*x3D) * ( fy * r[3] + cy * r[6] )  + (*y3D) * ( fy * r[4] + cy * r[7] ) + (*z3D) * (fy * r[5] +cy * r[8]);
  double scale =   t[2] + r[6] * (*x3D) + r[7] * (*y3D) + r[8] * (*z3D);

  if ( scale == 0.0 ) { fprintf(stderr,"could not projectPointsFrom3Dto2D"); return 0; }
  *x2D = x2DBuf / scale;
  *y2D = y2DBuf / scale;

 return 1;
}


int convertRodriguezAndTranslationTo4x4DUnprojectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit)
{
  double * matrix3x3Rotation = alloc4x4Matrix();    if (matrix3x3Rotation==0) { return 0; }

  //Our translation vector is ready to be used!
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);

  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((double*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  double * m  = result4x4;
  double * rm = matrix3x3Rotation;
  double * tm = translation;


  //double scaleToDepthUnit = 1000.0; //Convert Unit to milimeters
  double Tx = tm[0]*scaleToDepthUnit;
  double Ty = tm[1]*scaleToDepthUnit;
  double Tz = tm[2]*scaleToDepthUnit;


  /*
      Here what we want to do is generate a 4x4 matrix that does the inverse transformation that our
      rodriguez and translation vector define

      In order to do that we should have the following be true

                                      (note the minus under)
      (   R  |  T  )       (   R trans |  - R trans * T  )         (   I  |  0   )
      (  --------- )    .  (  -------------------------- )     =   ( ----------- )
      (   0  |  1  )       (   0       |        1        )         (   0  |  I   )

      Using matlab to do the calculations we get the following matrix
  */

   m[0]=  rm[0];        m[1]= rm[3];        m[2]=  rm[6];       m[3]= -1.0 * ( rm[0]*Tx + rm[3]*Ty + rm[6]*Tz );
   m[4]=  rm[1];        m[5]= rm[4];        m[6]=  rm[7];       m[7]= -1.0 * ( rm[1]*Tx + rm[4]*Ty + rm[7]*Tz );
   m[8]=  rm[2];        m[9]= rm[5];        m[10]= rm[8];       m[11]=-1.0 * ( rm[2]*Tx + rm[5]*Ty + rm[8]*Tz );
   m[12]= 0.0;          m[13]= 0.0;         m[14]=0.0;          m[15]=1.0;


  print4x4DMatrix("ModelView", result4x4);
  free4x4Matrix(&matrix3x3Rotation);
  return 1;
}


int convertRodriguezAndTranslationToOpenGL4x4DProjectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit )
{
  double * matrix3x3Rotation = alloc4x4Matrix();    if (matrix3x3Rotation==0) { return 0; }

  //Our translation vector is ready to be used!
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);

  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((double*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  double * m  = result4x4;
  double * rm = matrix3x3Rotation;
  double * tm = translation;


  //double scaleToDepthUnit = 1000.0; //Convert Unit to milimeters
  double Tx = tm[0]*scaleToDepthUnit;
  double Ty = tm[1]*scaleToDepthUnit;
  double Tz = tm[2]*scaleToDepthUnit;

  /*
      Here what we want to do is generate a 4x4 matrix that does the normal transformation that our
      rodriguez and translation vector define
  */
   m[0]=  rm[0];        m[1]= rm[1];        m[2]=  rm[2];       m[3]= -Tx;
   m[4]=  rm[3];        m[5]= rm[4];        m[6]=  rm[5];       m[7]= -Ty;
   m[8]=  rm[6];        m[9]= rm[7];        m[10]= rm[8];       m[11]=-Tz;
   m[12]= 0.0;          m[13]= 0.0;         m[14]=0.0;          m[15]=1.0;

  print4x4DMatrix("ModelView", result4x4);

  fprintf(stderr,"Matrix will be transposed to become OpenGL format ( i.e. column major )\n");
  transpose4x4MatrixD(result4x4);

  free4x4Matrix(&matrix3x3Rotation);
  return 1;
}


int move3DPoint(double * resultPoint3D, double * transformation4x4, double * point3D  )
{
  return transform3DPointUsing4x4Matrix(resultPoint3D,transformation4x4,point3D);
}



void buildOpenGLProjectionForIntrinsics   (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           )
{
   fprintf(stderr,"buildOpenGLProjectionForIntrinsics according to old Ammar code Image ( %u x %u )\n",imageWidth,imageHeight);
   fprintf(stderr,"fx %0.2f fy %0.2f , cx %0.2f , cy %0.2f , skew %0.2f \n",fx,fy,cx,cy,skew);
   fprintf(stderr,"Near %0.2f Far %0.2f \n",nearPlane,farPlane);


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    double L = 0.0 , B = 0.0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = nearPlane , F = farPlane;
    double R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N , F_plus_N = F+N , F_mul_N = F*N;

    if  ( (R_sub_L==0) || (R_sub_L-1.0f==0) ||
          (T_sub_B==0) || (T_sub_B-1.0f==0) ||
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limigs R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

   //OpenGL Projection Matrix ready for loading ( column-major ) , also axis compensated
   frustum[0] = -2.0f*fx/R_sub_L;     frustum[1] = 0.0f;                 frustum[2] = 0.0f;                              frustum[3] = 0.0f;
   frustum[4] = 0.0f;                 frustum[5] = 2.0f*fy/T_sub_B;      frustum[6] = 0.0f;                              frustum[7] = 0.0f;
   frustum[8] = 2.0f*cx/R_sub_L-1.0f; frustum[9] = 2.0f*cy/T_sub_B-1.0f; frustum[10]=-1.0*(F_plus_N/F_sub_N);            frustum[11] = -1.0f;
   frustum[12]= 0.0f;                 frustum[13]= 0.0f;                 frustum[14]=-2.0f*F_mul_N/(F_sub_N);            frustum[15] = 0.0f;
   //Matrix already in OpenGL column major format
}








void testMatrices()
{
   testPNPSolver();
  //testGJSolver();
  return ;


  double A[16]={ 1 ,2 ,3 ,4,
                 5 ,6 ,7 ,8,
                 9 ,10,11,12,
                 13,14,15,16
                };


  double B[16]={ 1 ,2 ,3 ,4,
                 4 ,3 ,2 ,1,
                 1 ,2 ,3 ,4,
                 4 ,3 ,2 ,1
                };

  double Res[16]={0};

  multiplyTwo4x4Matrices(Res,A,B);
/*
  28.000000 26.000000 24.000000 22.000000
  68.000000 66.000000 64.000000 62.000000
  108.000000 106.000000 104.000000 102.000000
  148.000000 146.000000 144.000000 142.000000*/

}
