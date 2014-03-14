#include <stdio.h>
#include <stdlib.h>

#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/solveLinearSystemGJ.h"

#include <gsl/gsl_matrix.h>

int main()
{
    /*
  int i, j;
  gsl_matrix * m = gsl_matrix_alloc (10, 3);

  for (i = 0; i < 10; i++)
    for (j = 0; j < 3; j++)
      gsl_matrix_set (m, i, j, 0.23 + 100*i + j);

  for (i = 0; i < 100; i++)  //OUT OF RANGE ERROR
    for (j = 0; j < 3; j++)
      printf ("m(%d,%d) = %g\n", i, j,
              gsl_matrix_get (m, i, j));

  gsl_matrix_free (m);*/


   //gsl_linalg_SV_decomp (gsl_matrix * A, gsl_matrix * V, gsl_vector * S, gsl_vector * work)

  printf("Testing Gauss-Jordan code..!\n");
     testGJSolver();
    return 0;
}
