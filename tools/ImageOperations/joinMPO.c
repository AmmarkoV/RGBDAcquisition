
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char * readFileToMemory(char * filename,unsigned int *length )
{
  *length = 0;
  FILE * pFile=0;

  char * buffer;
  size_t result;

  pFile = fopen ( filename , "rb" );
  if (pFile==0) { fprintf(stderr,"Could not read file %s \n",filename); return 0; }

  // obtain file size:
  fseek (pFile , 0 , SEEK_END);
  unsigned long lSize = ftell (pFile);
  rewind (pFile);

  // allocate memory to contain the whole file:
  buffer = (char*) malloc (sizeof(char)*lSize);
  if (buffer == 0 ) { fprintf(stderr,"Could not allocate enough memory for file %s ",filename); fclose(pFile); return 0; }

  // copy the file into the buffer:
  result = fread (buffer,1,lSize,pFile);
  if (result != lSize)
    {
      fprintf(stderr,"Could not read the whole file onto memory %s ",filename);
      fclose(pFile);
      free(buffer);
      return 0;
    }

  /* the whole file is now loaded in the memory buffer. */

  // terminate
  fclose (pFile);


  *length = lSize;
  return buffer;
}


int main(int argc, char** argv)
{
  char c;
  int views = 0;  // number of views (JPG components)
  long length;    // total length of file
  size_t amount;  // amount read
  char* buffer;
  char* leftFilename;
  char* rightFilename;
  char* outFilename;
  char* leftFilenameBase;
  char* rightFilenameBase;

     if (argc != 4) {
                       fprintf(stdout, "Usage: %s [filenameLeft.jpg] [filenameRight.jpg] [fileOutput.mpo] \n", argv[0]);
                       return 0;
                    } else
                    {
                      leftFilename = argv[1];
                      rightFilename = argv[2];
                      outFilename = argv[3];

                      leftFilenameBase = strdup(argv[1]);
                      rightFilenameBase = strdup(argv[1]);

                      char* ext = strstr(leftFilenameBase,".jpg");
                      if (ext != NULL) { ext[0] = '\0'; }
                            ext = strstr(rightFilenameBase,".jpg");
                      if (ext != NULL) { ext[0] = '\0'; }
                    }


  FILE* f = fopen(outFilename,"wb");
  if (f==NULL) {
                 fprintf(stderr,"error opening file \"%s\"\n",outFilename);
                 return 1;
               }

  //char magic[4]={ (char) 0xff , (char) 0xd8  , (char) 0xff , (char) 0xe1 };

//  fprintf(f,"%c%c%c%c",magic[0],magic[1],magic[2],magic[3]);


  char * jpegFile=0;
  unsigned int jpegFileSize=0;

  jpegFile = readFileToMemory(leftFilename,&jpegFileSize);
  fwrite(jpegFile,1,jpegFileSize,f);
  free(jpegFile);

//  fprintf(f,"%c%c%c%c",magic[0],magic[1],magic[2],magic[3]);
  jpegFile = readFileToMemory(rightFilename,&jpegFileSize);
  fwrite(jpegFile,1,jpegFileSize,f);
  free(jpegFile);

  fclose(f);

  return 0;
}

