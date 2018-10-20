#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include "../ModelLoader/model_loader.h"
#include "texture_loader.h"


#define USE_CODECS_LIBRARY 1


#if USE_CODECS_LIBRARY
 #include "../../../../../tools/Codecs/codecs.h"
#else
 #warning "NOT USING CODECS LIBRARY MEANS NO TEXTURES LOADED"
#endif // USE_CODECS_LIBRARY

// TODO HERE : CHECK FOR DUPLICATE TEXTURES , ETC ...

GLuint loadTexture(int type,char * directory ,char *fname)
{
  #if USE_CODECS_LIBRARY
   fprintf(stderr,"Using Codecs Library to load texture %s \n",fname);
  #else
   fprintf(stderr,"These library is compiled without the codecs Library\n Cannot load texture %s \n",fname);
   return 0;
  #endif // USE_CODECS_LIBRARY

    char fullPath[MAX_MODEL_PATHS*2 + 2 ]={0};
    strncpy(fullPath,directory,MAX_MODEL_PATHS);
    strcat(fullPath,"/");
    strncat(fullPath,fname,MAX_MODEL_PATHS);

    fprintf(stderr,"Making Texture , Type %u , Name %s , Path %s \n",type , fname , fullPath);

	GLuint tex=0;
	GLubyte *bits;
	unsigned int width=0 , height=0 , bitsperpixel=0 , channels=0;

    bits = ( GLubyte * ) readImageRaw(fullPath,0 /*AUTO*/,&width,&height,&bitsperpixel,&channels);

    glGenTextures(1,&tex);
	glBindTexture(GL_TEXTURE_2D,tex);
	// define what happens if given (s,t) outside [0,1] {REPEAT, CLAMP}

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, type);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, type);
	//glTexImage2D ( GL_TEXTURE_2D, 0, 3, width, height , 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, bits);

	glTexImage2D ( GL_TEXTURE_2D, 0 /*BASE IMAGE LEVEL */,GL_RGB, width, height , 0,GL_RGB, GL_UNSIGNED_BYTE, bits);

    //mip mapping
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST);
    //gluBuild2DMipmaps( GL_TEXTURE_2D, 3,
	//                   info->bmiHeader.biWidth, info->bmiHeader.biHeight,
	//			         GL_BGR_EXT, GL_UNSIGNED_BYTE, bits );

    free(bits);
    fprintf(stderr,"Survived and made texture %u ",tex);
	return tex;
}
