/* A small tool for splitting MPO files into their JPG components.
 * $Id: mposplit.c,v 1.5 2012/06/24 01:16:33 chris Exp $
 * Copyright (C) 2009-2012, Christian Steinruecken. All rights reserved.
 * 
 * This code is released under the Revised BSD Licence.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   - The names of the author(s) and contributors may not be used to
 *     endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * DISCLAIMER:
 * This software is provided by the copyright holders and contributors
 * "as is" and any express or implied warranties, including, but not
 * limited to, the implied warranties of merchantability and fitness for
 * a particular purpose are disclaimed.  In no event shall the copyright
 * holders be liable for any direct, indirect, incidental, special,
 * exemplary, or consequential damages (including, but not limited to,
 * procurement of substitute goods or services; loss of use, data, or
 * profits; or business interruption) however caused and on any theory of
 * liability, whether in contract, strict liability, or tort (including
 * negligence or otherwise) arising in any way out of the use of this
 * software, even if advised of the possibility of such damage. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char** argv) {
  char c;
  int views = 0;  // number of views (JPG components)
  long length;    // total length of file
  size_t amount;  // amount read
  char* buffer;
  char* fnm;
  char* fnmbase;
  if (argc != 2) {
    fprintf(stdout, "Usage: %s [filename]\n", argv[0]);
    return 0;
  } else {
    fnm = argv[1];
    fnmbase = strdup(argv[1]);
    char* ext = strstr(fnmbase,".mpo");
    if (ext != NULL) {
      ext[0] = '\0';
    }
  }

  FILE* f = fopen(fnm,"rb");
  if (f==NULL) {
    fprintf(stderr,"error opening file \"%s\"\n",fnm);
    return 1;
  }
  // obtain file size:
  fseek(f, 0, SEEK_END);
  length = ftell(f);
  rewind(f);

  // allocate memory to contain the whole file:
  //char buffer[BUFSIZ];
  buffer = (char*) malloc (sizeof(char)*length);
  if (buffer == NULL) {
    fprintf(stderr,"failed to allocate memory\n");
    return 2;
  } else {
    fprintf(stdout,"Allocated %ld chars of memory.\n",length);
  }
  amount = fread(buffer,1,length,f);
  if (amount != length) {
    fprintf(stderr,"error loading file\n");
    return 3;
  }
  fclose(f);
  // NOW find the individual JPGs...
  char* view = buffer;
  char* last = NULL;
  char* wnm = (char*) malloc(256);
  //fprintf(stdout,"Started at %p.\n",view);
  while (view < buffer+length-4) {
    if (((char) view[0] % 255) == (char) 0xff) {
      if (((char) view[1] % 255) == (char) 0xd8) {
        if (((char) view[2] % 255) == (char) 0xff) {
          if (((char) view[3] % 255) == (char) 0xe1) {
            fprintf(stdout, "View found at offset %d\n", view-buffer);
            views++;
            if (last != NULL) {
              // copy out the previous view
              sprintf(wnm, "%s.v%d.jpg", fnmbase, views-1);
              FILE* w = fopen(wnm, "wb");
              fwrite(last, 1, view-last, w);
              fclose(w);
              fprintf(stdout, "Created %s\n",wnm);
            }
            last = view;
            view+=4;
          } else {
            view+=2;
          }
        } else {
          view+=3;
        }
      } else {
        view+=1;
      }
    } else {
      view+=1;
    }
  }
  //fprintf(stdout,"Stopped at %p.\n",view);
  if (views > 1) {
    // copy out the last view
    sprintf(wnm, "%s.v%d.jpg", fnmbase, views);
    FILE* w = fopen(wnm, "wb");
    fwrite(last, 1, buffer+length-last, w);
    fclose(w);
    fprintf(stdout, "Created %s\n",wnm);
  } else
  if (views == 0) {
    fprintf(stdout, "No views found.\n");
  }
  free(wnm);
  free(buffer);
  return 0;
}

