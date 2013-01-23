#include "save_to_file.h"
#include <stdio.h>
#include <stdlib.h>

void WriteOpenGLColor(char * depthfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height)
{

    char * zbuffer = (char *) malloc((width-x)*(height-y)*sizeof(char)*3);

    getOpenGLColor(zbuffer, x, y, width,  height);
    saveRawImageToFile(depthfile,zbuffer,(width-x),(height-y),3,8);

    if (zbuffer!=0) { free(zbuffer); zbuffer=0; }

    return ;
}
