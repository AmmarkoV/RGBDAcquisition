#include <stdio.h>
#include <stdlib.h>

#include "fastStringParser.h"


int main(int argc, char *argv[])
{
  if (argc<1) { fprintf(stderr,"Please add a filename string as a parameter\n"); return 1; }

  struct fastStringParser * fsp = fastSTringParser_createRulesFromFile(argv[1],64);

  export_C_Scanner(fsp,argv[1]);

  return 0;
}
