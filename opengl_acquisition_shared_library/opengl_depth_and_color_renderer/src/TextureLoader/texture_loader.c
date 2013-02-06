#include <stdio.h>
#include <stdlib.h>
#include <GL/gl.h>
#include <GL/glx.h>    /* this includes the necessary X headers */

#include "texture_loader.h"
#include "bmp.h"
#include "ppm.h"



// TODO HERE : CHECK FOR DUPLICATE TEXTURES , ETC ...


GLuint loadTexture(int type,const char *fname)
{
    fprintf(stderr,"Making Texture , Type %u , Name %s \n",type , fname);

	GLuint tex=0;
	GLubyte *bits;
	unsigned int width=0;
	unsigned int height=0;


    /* BMP LOADER HERE
	BITMAPINFO *info;
	bits=LoadDIBitmap(fname,&info);
    if (bits==0) { printf("Cannot Make Texture of %s \n",fname); return 0;}
    width = info->bmiHeader.biWidth ;
    height = info->bmiHeader.biHeight;
    */


    //PPM LOADER
    bits = ( GLubyte * ) ReadPPM((char *) fname,&width,&height);

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









// load a 256x256 RGB .RAW file as a texture
GLuint LoadTextureRAW( const char * filename, int wrap )
{
  GLuint texture;
  int width, height;
  unsigned char * data;
  FILE * file;

  // open texture data
  file = fopen( filename, "rb" );
  if ( file == NULL ) return 0;

  // allocate buffer
  width = 256;
  height = 256;
  data = malloc( width * height * 3 );

  // read texture data
  fread( data, width * height * 3, 1, file );
  fclose( file );

  // allocate a texture name
  glGenTextures( 1, &texture );

  // select our current texture
  glBindTexture( GL_TEXTURE_2D, texture );

  // select modulate to mix texture with color for shading
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

  // when texture area is small, bilinear filter the closest MIP map
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                   GL_LINEAR_MIPMAP_NEAREST );
  // when texture area is large, bilinear filter the first MIP map
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

  // if wrap is true, the texture wraps over at the edges (repeat)
  //       ... false, the texture ends at the edges (clamp)
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                   wrap ? GL_REPEAT : GL_CLAMP );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                   wrap ? GL_REPEAT : GL_CLAMP );

  // build our texture MIP maps

	glTexImage2D ( GL_TEXTURE_2D, 0, 3, width, height , 0, GL_RGB, GL_UNSIGNED_BYTE, data);

  // free buffer
  free( data );

  return texture;

}









