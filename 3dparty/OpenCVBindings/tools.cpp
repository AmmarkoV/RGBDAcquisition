#include "tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void clear_line()
{
  fputs("\033[A\033[2K\033[A\033[2K",stdout);
  rewind(stdout);
  int i=ftruncate(1,0);
  if (i!=0) { /*fprintf(stderr,"Error with ftruncate\n");*/ }
}
