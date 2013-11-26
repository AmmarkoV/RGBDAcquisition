#include <stdio.h>
#include <stdlib.h>

#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/solveLinearSystemGJ.h"

int main()
{
    printf("Testing Gauss-Jordan code..!\n");
     testGJSolver();
    return 0;
}
