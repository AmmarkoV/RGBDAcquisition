/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 * ( code has been modified , original file found here http://zarb.org/~gc/html/libpng.html
 */
#define USE_PNG_FILES 1


#include "pngInput.h"

#if USE_PNG_FILES

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define PNG_DEBUG 3
#include <png.h>

void abort_(const char * s, ...)
{
        va_list args;
        va_start(args, s);
        vfprintf(stderr, s, args);
        fprintf(stderr, "\n");
        va_end(args);
       // abort();
}


int ReadPNG(char *filename,struct Image * pic,char read_only_header)
{
    png_byte header[8];

    FILE *fp = fopen(filename, "rb");
    if (fp == 0)
    {
        perror(filename);
        return 0;
    }

    // read the header
    fread(header, 1, 8, fp);

    if (png_sig_cmp(header, 0, 8))
    {
        fprintf(stderr, "error: %s is not a PNG.\n", filename);
        fclose(fp);
        return 0;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        fprintf(stderr, "error: png_create_read_struct returned 0.\n");
        fclose(fp);
        return 0;
    }

    // create png info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        fprintf(stderr, "error: png_create_info_struct returned 0.\n");
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        fclose(fp);
        return 0;
    }

    // create png info struct
    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info)
    {
        fprintf(stderr, "error: png_create_info_struct returned 0.\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) NULL);
        fclose(fp);
        return 0;
    }

    // the code in this if statement gets called if libpng encounters an error
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "error from libpng\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return 0;
    }

    // init png reading
    png_init_io(png_ptr, fp);

    // let libpng know you already read the first 8 bytes
    png_set_sig_bytes(png_ptr, 8);

    // read all the info up to the image data
    png_read_info(png_ptr, info_ptr);

    pic->width = png_get_image_width(png_ptr, info_ptr);
    pic->height = png_get_image_height(png_ptr, info_ptr);
    pic->channels = png_get_channels(png_ptr, info_ptr);
    pic->bitsperpixel = png_get_bit_depth(png_ptr, info_ptr);

    // variables to pass to get info
    int bit_depth, color_type;
    png_uint_32 temp_width, temp_height;

    // get info about png
    png_get_IHDR(png_ptr, info_ptr, &temp_width, &temp_height, &bit_depth, &color_type,
        NULL, NULL, NULL);


    // Update the png info struct.
    png_read_update_info(png_ptr, info_ptr);

    // Row size in bytes.
    int rowbytes = png_get_rowbytes(png_ptr, info_ptr);

    // glTexImage2d requires rows to be 4-byte aligned
    rowbytes += 3 - ((rowbytes-1) % 4);

    // Allocate the image_data as a big block, to be given to opengl
    png_byte * image_data;
    image_data = malloc(rowbytes * temp_height * sizeof(png_byte)+15);
    if (image_data == NULL)
    {
        fprintf(stderr, "error: could not allocate memory for PNG image data\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return 0;
    }

    // row_pointers is for pointing to image_data for reading the png with libpng
    png_bytep * row_pointers = malloc(temp_height * sizeof(png_bytep));
    if (row_pointers == NULL)
    {
        fprintf(stderr, "error: could not allocate memory for PNG row pointers\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        free(image_data);
        fclose(fp);
        return 0;
    }

    // set the individual row_pointers to point at the correct offsets of image_data
    int i;
    for (i = 0; i < temp_height; i++)
    {
//  INVERT Y row_pointers[pic->height - 1 - i] = image_data + i * rowbytes;
             row_pointers[i] = image_data + i * rowbytes;
    }

    // read the png into image_data through row_pointers
    png_read_image(png_ptr, row_pointers);


        pic->pixels = image_data;

    // clean up
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    free(row_pointers);
    fclose(fp);

    return 1;
}


int WritePNG(char * filename,struct Image * pic)
{

// VARIABLES START ---------------
png_byte color_type;
png_byte bit_depth;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;

int width, height;
// -------------------------------


/* create file */
FILE *fp = fopen(filename, "wb");
if (!fp) { abort_("[write_png_file] File %s could not be opened for writing", filename); return 0; }


/* initialize stuff */
png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
if (!png_ptr) { abort_("[write_png_file] png_create_write_struct failed"); return 0; }

info_ptr = png_create_info_struct(png_ptr);
if (!info_ptr) { abort_("[write_png_file] png_create_info_struct failed"); return 0; }

if (setjmp(png_jmpbuf(png_ptr))) { abort_("[write_png_file] Error during init_io"); return 0; }

png_init_io(png_ptr, fp);


/* write header */
if (setjmp(png_jmpbuf(png_ptr))) { abort_("[write_png_file] Error during writing header"); return 0; }

png_set_IHDR(png_ptr, info_ptr, pic->width,pic->height,8,PNG_COLOR_TYPE_RGB , PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
png_write_info(png_ptr, info_ptr);


/* write bytes */
if (setjmp(png_jmpbuf(png_ptr))) { abort_("[write_png_file] Error during writing bytes"); return 0; }
row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * pic->height);
if (row_pointers==0) { abort_("Could not allocate enough memory to hold the image \n"); return 0; }
char * raw_pixels=pic->pixels;

unsigned int y;
for (y=0; y<pic->height; y++)
 {
   row_pointers[y] = (png_byte*) raw_pixels;
   raw_pixels+=3*pic->width;
 }

 png_write_image(png_ptr,row_pointers);


 /* end write */
 if (setjmp(png_jmpbuf(png_ptr))) { abort_("[write_png_file] Error during end of write"); return 0; }

 png_write_end(png_ptr, NULL);

 /* cleanup heap allocation */
 free(row_pointers);

 fclose(fp);
return 1;
}

#endif
