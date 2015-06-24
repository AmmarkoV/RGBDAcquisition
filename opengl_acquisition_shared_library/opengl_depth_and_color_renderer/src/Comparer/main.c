#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    printf("Hello world!\n");


  int i=0;
  for (i=0; i<argc; i++)
  {
     if (strcmp(argv[i],"-from1")==0) { } else
     if (strcmp(argv[i],"-from2")==0) {  }

  }


    return 0;
}
