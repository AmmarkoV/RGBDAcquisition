#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include "../ModelLoader/model_loader.h"
#include "texture_loader.h"
#include "bmp.h"
#include "ppm.h"



// TODO HERE : CHECK FOR DUPLICATE TEXTURES , ETC ...


GLuint loadTexture(int type,char * directory ,char *fname)
{
    char fullPath[MAX_MODEL_PATHS*2 + 2 ]={0};
    strncpy(fullPath,directory,MAX_MODEL_PATHS);
    strcat(fullPath,"/");
    strncat(fullPath,fname,MAX_MODEL_PATHS);

    fprintf(stderr,"Making Texture , Type %u , Name %s , Path %s \n",type , fname , fullPath);

	GLuint tex=0;
	GLubyte *bits;
	unsigned int width=0;
	unsigned int height=0;


  if ( strstr(fname,".bmp") != 0 )
    {
      //BMP LOADER HERE
	  BITMAPINFO *info;
	  bits=LoadDIBitmap(fname,&info);
      if (bits==0) { printf("Cannot Make Texture of %s \n",fname); return 0;}
      width = info->bmiHeader.biWidth ;
      height = info->bmiHeader.biHeight;
    }
     else
  if ( strstr(fname,".ppm") != 0 )
    {
      //PPM LOADER
      bits = ( GLubyte * ) ReadPPM((char *) fullPath,&width,&height);
    }


    glGenTextures(1,&tex);
	glBindTexture(GL_TEXTURE_2D,tex);
	// define what happens if given (s,t) outside [0,1] {REPEAT, CLAMP}

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, type);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, type);
	//glTexImage2D ( GL_TEXTURE_2D, 0, 3, width, height , 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, bits);

	glTexImage2D ( GL_TEXTURE_2D, 0, 3, width, height , 0, GL_RGB, GL_UNSIGNED_BYTE, bits);



    // mip mapping
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST);
//	gluBuild2DMipmaps( GL_TEXTURE_2D, 3,
	//	               info->bmiHeader.biWidth, info->bmiHeader.biHeight,
		//			   GL_BGR_EXT, GL_UNSIGNED_BYTE, bits );



    free(bits);
    fprintf(stderr,"Survived and made texture %u ",tex);
	return tex;
}








