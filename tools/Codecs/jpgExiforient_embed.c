/*
 * jpegexiforient.c , transformed to embed it in flashy slideshows..
 *
 * This is a utility program to get and set the Exif Orientation Tag.
 * It can be used together with jpegtran in scripts for automatic
 * orientation correction of digital camera pictures.
 *
 * The Exif orientation value gives the orientation of the camera
 * relative to the scene when the image was captured.  The relation
 * of the '0th row' and '0th column' to visual position is shown as
 * below.
 *
 * Value | 0th Row     | 0th Column
 * ------+-------------+-----------
 *   1   | top         | left side
 *   2   | top         | right side
 *   3   | bottom      | right side
 *   4   | bottom      | left side
 *   5   | left side   | top
 *   6   | right side  | top
 *   7   | right side  | bottom
 *   8   | left side   | bottom
 *
 * For convenience, here is what the letter F would look like if it were
 * tagged correctly and displayed by a program that ignores the orientation
 * tag:
 *
 *   1        2       3      4         5            6           7          8
 *
 * 888888  888888      88  88      8888888888  88                  88  8888888888
 * 88          88      88  88      88  88      88  88          88  88      88  88
 * 8888      8888    8888  8888    88          8888888888  8888888888          88
 * 88          88      88  88
 * 88          88  888888  888888
 *
 */
#include "jpgExiforient_embed.h"
#include <stdio.h>
#include <stdlib.h>

static FILE * myfile;		/* My JPEG file */
static int read_byte_error_while_reading=0;		/* My JPEG file */
static unsigned char exif_data[65536L];

/* Return next input byte, or EOF if no more */
#define NEXTBYTE()  getc(myfile)

/* Error exit handler */
#define ERREXIT(msg)  (exit(0))

/* Read one byte, testing for EOF */
static int
read_1_byte (void)
{
  int c;

  c = NEXTBYTE();
  if (c == EOF) { read_byte_error_while_reading=1; return 0; }
                //ERREXIT("Premature EOF in JPEG file");
  return c;
}

/* Read 2 bytes, convert to unsigned int */
/* All 2-byte quantities in JPEG markers are MSB first */
static unsigned int
read_2_bytes (void)
{
  int c1, c2;

  c1 = NEXTBYTE();
  if (c1 == EOF)  { read_byte_error_while_reading=1; return 0; }
                  //ERREXIT("Premature EOF in JPEG file");
  c2 = NEXTBYTE();
  if (c2 == EOF)  { read_byte_error_while_reading=1; return 0; }
                  //ERREXIT("Premature EOF in JPEG file");
  return (((unsigned int) c1) << 8) + ((unsigned int) c2);
}


/*
 * The main program.
 */

unsigned int GetJPGOrientationOfFile(char * filename)
{
  read_byte_error_while_reading=0;
  myfile=0;
  int set_flag=0;
  unsigned int length=0, i=0;
  int is_motorola=0; /* Flag for byte order */
  unsigned int offset=0, number_of_tags=0, tagnum=0;



   set_flag = 0;

  i = 1;
  myfile = fopen(filename, "rb");
    if ( myfile == 0)
    {
      fprintf(stderr, " Can't open %s\n", filename);
      return 0;
    }

  /* Read File head, check for JPEG SOI + Exif APP1 */
  for (i = 0; i < 4; i++) { exif_data[i] = (unsigned char) read_1_byte(); }

  if (exif_data[0] != 0xFF ||
      exif_data[1] != 0xD8 ||
      exif_data[2] != 0xFF ||
      exif_data[3] != 0xE1) { fclose(myfile); return 0; }

  /* Get the marker parameter length count */
  length = read_2_bytes();
  /* Length includes itself, so must be at least 2 */
  /* Following Exif data length must be at least 6 */
  if (length < 8) { fclose(myfile); return 0; }
  length -= 8;
  /* Read Exif head, check for "Exif" */
  for (i = 0; i < 6; i++) { exif_data[i] = (unsigned char) read_1_byte(); }

  if (exif_data[0] != 0x45 ||
      exif_data[1] != 0x78 ||
      exif_data[2] != 0x69 ||
      exif_data[3] != 0x66 ||
      exif_data[4] != 0 ||
      exif_data[5] != 0) { fclose(myfile); return 0; }

  /* Read Exif body */
  for (i = 0; i < length; i++) { exif_data[i] = (unsigned char) read_1_byte(); }


  if (length < 12) { fclose(myfile); return 0; } /* Length of an IFD entry */

  /* Discover byte order */
  if (exif_data[0] == 0x49 && exif_data[1] == 0x49) { is_motorola = 0; } else
  if (exif_data[0] == 0x4D && exif_data[1] == 0x4D) { is_motorola = 1; } else
                                                    { fclose(myfile); return 0; }

  /* Check Tag Mark */
  if (is_motorola)
  {
    if (exif_data[2] != 0)    { fclose(myfile); return 0; }
    if (exif_data[3] != 0x2A) { fclose(myfile); return 0; }
  } else
  {
    if (exif_data[3] != 0)    { fclose(myfile); return 0; }
    if (exif_data[2] != 0x2A) { fclose(myfile); return 0; }
  }

  /* Get first IFD offset (offset to IFD0) */
  if (is_motorola)
  {
    if (exif_data[4] != 0) { fclose(myfile); return 0; }
    if (exif_data[5] != 0) { fclose(myfile); return 0; }
    offset = exif_data[6];
    offset <<= 8;
    offset += exif_data[7];
  } else
  {
    if (exif_data[7] != 0) { fclose(myfile); return 0; }
    if (exif_data[6] != 0) { fclose(myfile); return 0; }
    offset = exif_data[5];
    offset <<= 8;
    offset += exif_data[4];
  }

  if (offset > length - 2) { fclose(myfile); return 0; } /* check end of data segment */

  /* Get the number of directory entries contained in this IFD */
  if (is_motorola)
  {
    number_of_tags = exif_data[offset];
    number_of_tags <<= 8;
    number_of_tags += exif_data[offset+1];
  } else
  {
    number_of_tags = exif_data[offset+1];
    number_of_tags <<= 8;
    number_of_tags += exif_data[offset];
  }
  if (number_of_tags == 0) { fclose(myfile); return 0; }
  offset += 2;

  /* Search for Orientation Tag in IFD0 */
  for (;;)
  {
    if (offset > length - 12) { fclose(myfile); return 0; }/* check end of data segment */
    /* Get Tag number */
    if (is_motorola)
    {
      tagnum = exif_data[offset];
      tagnum <<= 8;
      tagnum += exif_data[offset+1];
    } else
    {
      tagnum = exif_data[offset+1];
      tagnum <<= 8;
      tagnum += exif_data[offset];
    }
    if (tagnum == 0x0112) break; /* found Orientation Tag */
    if (--number_of_tags == 0) { fclose(myfile); return 0; }
    offset += 12;
  }



    /* Get the Orientation value */
    if (is_motorola)
    {
      if (exif_data[offset+8] != 0) { fclose(myfile); return 0; }
      set_flag = exif_data[offset+9];
    } else
    {
      if (exif_data[offset+9] != 0) { fclose(myfile); return 0; }
      set_flag = exif_data[offset+8];
    }
    if (set_flag > 8) { fclose(myfile); return 0; }

    fclose(myfile);
  /* All done. */
  return (unsigned int) set_flag;
}
