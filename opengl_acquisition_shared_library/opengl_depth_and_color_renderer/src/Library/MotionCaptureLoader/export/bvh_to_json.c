#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_to_json.h"

#include "bvh_export.h"

#include "../bvh_loader.h"

#include "../calculate/bvh_project.h"
#include "../edit/bvh_remapangles.h"

#define CONVERT_EULER_TO_RADIANS M_PI/180.0
#define DUMP_SEPERATED_POS_ROT 0
#define DUMP_3D_POSITIONS 0


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

/*
unsigned int invisibleJoints=0;
unsigned int   visibleJoints=0;
unsigned int filteredOutCSVBehindPoses=0;
unsigned int filteredOutCSVOutPoses=0;
unsigned int filteredOutCSVPoses=0;*/
