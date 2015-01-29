#include <stdio.h>
#include <stdlib.h>
#include "patterns.h"

struct pattern patA={0};
struct pattern patB={0};

int main(int argc, char *argv[])
{
    printf("Checking for patterns from first two inputs\n");

    //----------------------------------------------------------
    fprintf(stderr,"\n\n");
    //----------------------------------------------------------

    convertStringToPattern(&patA, argv[1]);
    viewPattern(&patA , "Prototype Initial (A)");

    convertStringToPattern(&patB, argv[2]);
    viewPattern(&patB , "Observed Initial (B)");

    //----------------------------------------------------------
    fprintf(stderr,"\n\n");
    //----------------------------------------------------------

    cleanPattern(&patA,0.0);
    viewPattern(&patA , "Prototype (A)");

    cleanPattern(&patB,0.0);
    viewPattern(&patB , "Observed (B)");

    //----------------------------------------------------------

    if (!patternsMatch(&patA,&patB)) { fprintf(stderr,"Mismatch\n"); return 1; }

    fprintf(stderr,"Match\n");
    return 0;
}
