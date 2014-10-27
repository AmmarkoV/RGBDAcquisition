#include <stdio.h>
#include <stdlib.h>
#include "patterns.h"

struct pattern patA={0};
struct pattern patB={0};

int main(int argc, char *argv[])
{
    printf("Checking for patterns from first two inputs\n");


    convertStringToPattern(&patA, argv[1]);
    convertStringToPattern(&patB, argv[2]);

    cleanPattern(&patA,0.2);
    viewPattern(&patA , "Pattern A");

    cleanPattern(&patB,0.2);
    viewPattern(&patB , "Pattern B");


    if (!patternsMatch(&patA,&patB)) { fprintf(stderr,"Mismatch\n"); return 1; }

    fprintf(stderr,"Match\n");
    return 0;
}
