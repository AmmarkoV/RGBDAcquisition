
/*                  
This file was automatically generated @ 30-10-2019 12:52:41 using StringRecognizer                  
https://github.com/AmmarkoV/AmmarServer/tree/master/src/StringRecognizer                 
Please note that changes you make here may be automatically overwritten                  
if the String Recognizer generator runs again..!              
 */ 

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "TrajectoryPrimitives.h"

int scanFor_TrajectoryPrimitives(const char * str,unsigned int strLength) 
{
 if (str==0) { return 0; } 
 if (strLength<1) { return 0; } 

 switch (toupper(str[0])) { 
 case 'A' : 
     switch (toupper(str[1])) { 
     case 'F' : 
         if (strLength<32) { return 0; } 
         if ( strncasecmp(str,"AFFIX_OBJ_TO_OBJ_FOR_NEXT_FRAMES",32) == 0 ) { return TRAJECTORYPRIMITIVES_AFFIX_OBJ_TO_OBJ_FOR_NEXT_FRAMES; } 
     break; 
     case 'L' : 
         if (strLength<22) { return 0; } 
         if ( strncasecmp(str,"ALWAYS_SHOW_LAST_FRAME",22) == 0 ) { return TRAJECTORYPRIMITIVES_ALWAYS_SHOW_LAST_FRAME; } 
     break; 
     case 'R' : 
         if (strLength<5) { return 0; } 
         if ( strncasecmp(str,"ARROW",5) == 0 ) { return TRAJECTORYPRIMITIVES_ARROW; } 
     break; 
     case 'U' : 
         if (strLength<11) { return 0; } 
         if ( strncasecmp(str,"AUTOREFRESH",11) == 0 ) { return TRAJECTORYPRIMITIVES_AUTOREFRESH; } 
     break; 
    }; 
 break; 
 case 'B' : 
     if (strLength<10) { return 0; } 
     if ( strncasecmp(str,"BACKGROUND",10) == 0 ) { return TRAJECTORYPRIMITIVES_BACKGROUND; } 
 break; 
 case 'C' : 
     switch (toupper(str[1])) { 
     case 'O' : 
         switch (toupper(str[2])) { 
         case 'M' : 
             if ( (strLength >= 7 )&& ( strncasecmp(str,"COMMENT",7) == 0 ) ) { return TRAJECTORYPRIMITIVES_COMMENT; } 
             else  if ( (strLength >= 16 )&& ( strncasecmp(str,"COMPOSITE_OBJECT",16) == 0 ) ) { return TRAJECTORYPRIMITIVES_COMPOSITE_OBJECT; } 
         break; 
         case 'N' : 
             if ( (strLength >= 9 )&& ( strncasecmp(str,"CONNECTOR",9) == 0 ) ) { return TRAJECTORYPRIMITIVES_CONNECTOR; } 
         break; 
        }; 
     break; 
    }; 
 break; 
 case 'D' : 
     switch (toupper(str[1])) { 
     case 'E' : 
         if (strLength<5) { return 0; } 
         if ( strncasecmp(str,"DEBUG",5) == 0 ) { return TRAJECTORYPRIMITIVES_DEBUG; } 
     break; 
     case 'O' : 
         if (strLength<22) { return 0; } 
         if ( strncasecmp(str,"DONE_DECLARING_OBJECTS",22) == 0 ) { return TRAJECTORYPRIMITIVES_DONE_DECLARING_OBJECTS; } 
     break; 
    }; 
 break; 
 case 'E' : 
     switch (toupper(str[1])) { 
     case 'M' : 
         if (strLength<25) { return 0; } 
         if ( strncasecmp(str,"EMULATE_PROJECTION_MATRIX",25) == 0 ) { return TRAJECTORYPRIMITIVES_EMULATE_PROJECTION_MATRIX; } 
     break; 
     case 'V' : 
         if (strLength<5) { return 0; } 
         if ( strncasecmp(str,"EVENT",5) == 0 ) { return TRAJECTORYPRIMITIVES_EVENT; } 
     break; 
    }; 
 break; 
 case 'F' : 
     switch (toupper(str[1])) { 
     case 'A' : 
         if (strLength<8) { return 0; } 
         if ( strncasecmp(str,"FAR_CLIP",8) == 0 ) { return TRAJECTORYPRIMITIVES_FAR_CLIP; } 
     break; 
     case 'O' : 
         if (strLength<3) { return 0; } 
         if ( strncasecmp(str,"FOG",3) == 0 ) { return TRAJECTORYPRIMITIVES_FOG; } 
     break; 
     case 'R' : 
         switch (toupper(str[2])) { 
         case 'A' : 
             if ( (strLength >= 11 )&& ( strncasecmp(str,"FRAME_RESET",11) == 0 ) ) { return TRAJECTORYPRIMITIVES_FRAME_RESET; } 
             else  if ( (strLength >= 5 )&& ( strncasecmp(str,"FRAME",5) == 0 ) ) { return TRAJECTORYPRIMITIVES_FRAME; } 
         break; 
        }; 
     break; 
    }; 
 break; 
 case 'G' : 
     if (strLength<22) { return 0; } 
     if ( strncasecmp(str,"GENERATE_ANGLE_OBJECTS",22) == 0 ) { return TRAJECTORYPRIMITIVES_GENERATE_ANGLE_OBJECTS; } 
 break; 
 case 'H' : 
     if (strLength<11) { return 0; } 
     if ( strncasecmp(str,"HAND_POINTS",11) == 0 ) { return TRAJECTORYPRIMITIVES_HAND_POINTS; } 
 break; 
 case 'I' : 
     switch (toupper(str[1])) { 
     case 'N' : 
         switch (toupper(str[2])) { 
         case 'C' : 
             if ( (strLength >= 7 )&& ( strncasecmp(str,"INCLUDE",7) == 0 ) ) { return TRAJECTORYPRIMITIVES_INCLUDE; } 
         break; 
         case 'T' : 
             if ( (strLength >= 16 )&& ( strncasecmp(str,"INTERPOLATE_TIME",16) == 0 ) ) { return TRAJECTORYPRIMITIVES_INTERPOLATE_TIME; } 
         break; 
        }; 
     break; 
    }; 
 break; 
 case 'L' : 
     if (strLength<5) { return 0; } 
     if ( strncasecmp(str,"LIGHT",5) == 0 ) { return TRAJECTORYPRIMITIVES_LIGHT; } 
 break; 
 case 'M' : 
     switch (toupper(str[1])) { 
     case 'A' : 
         if (strLength<13) { return 0; } 
         if ( strncasecmp(str,"MAP_ROTATIONS",13) == 0 ) { return TRAJECTORYPRIMITIVES_MAP_ROTATIONS; } 
     break; 
     case 'O' : 
         switch (toupper(str[2])) { 
         case 'D' : 
             if ( (strLength >= 16 )&& ( strncasecmp(str,"MODELVIEW_MATRIX",16) == 0 ) ) { return TRAJECTORYPRIMITIVES_MODELVIEW_MATRIX; } 
         break; 
         case 'V' : 
             if ( (strLength >= 9 )&& ( strncasecmp(str,"MOVE_VIEW",9) == 0 ) ) { return TRAJECTORYPRIMITIVES_MOVE_VIEW; } 
             else  if ( (strLength >= 4 )&& ( strncasecmp(str,"MOVE",4) == 0 ) ) { return TRAJECTORYPRIMITIVES_MOVE; } 
         break; 
        }; 
     break; 
    }; 
 break; 
 case 'N' : 
     if (strLength<9) { return 0; } 
     if ( strncasecmp(str,"NEAR_CLIP",9) == 0 ) { return TRAJECTORYPRIMITIVES_NEAR_CLIP; } 
 break; 
 case 'O' : 
     switch (toupper(str[1])) { 
     case 'B' : 
         switch (toupper(str[2])) { 
         case 'J' : 
             if ( (strLength >= 21 )&& ( strncasecmp(str,"OBJECT_ROTATION_ORDER",21) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJECT_ROTATION_ORDER; } 
             else  if ( (strLength >= 11 )&& ( strncasecmp(str,"OBJECT_TYPE",11) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJECT_TYPE; } 
             else  if ( (strLength >= 10 )&& ( strncasecmp(str,"OBJECTTYPE",10) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJECTTYPE; } 
             else  if ( (strLength >= 6 )&& ( strncasecmp(str,"OBJECT",6) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJECT; } 
             else  if ( (strLength >= 10 )&& ( strncasecmp(str,"OBJ_OFFSET",10) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJ_OFFSET; } 
             else  if ( (strLength >= 3 )&& ( strncasecmp(str,"OBJ",3) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJ; } 
         break; 
        }; 
     break; 
     case 'F' : 
         if (strLength<16) { return 0; } 
         if ( strncasecmp(str,"OFFSET_ROTATIONS",16) == 0 ) { return TRAJECTORYPRIMITIVES_OFFSET_ROTATIONS; } 
     break; 
    }; 
 break; 
 case 'P' : 
     switch (toupper(str[1])) { 
     case 'O' : 
         switch (toupper(str[2])) { 
         case 'S' : 
             if ( (strLength >= 19 )&& ( strncasecmp(str,"POSE_ROTATION_ORDER",19) == 0 ) ) { return TRAJECTORYPRIMITIVES_POSE_ROTATION_ORDER; } 
             else  if ( (strLength >= 7 )&& ( strncasecmp(str,"POSERAW",7) == 0 ) ) { return TRAJECTORYPRIMITIVES_POSERAW; } 
             else  if ( (strLength >= 7 )&& ( strncasecmp(str,"POSE4x4",7) == 0 ) ) { return TRAJECTORYPRIMITIVES_POSE4X4; } 
             else  if ( (strLength >= 5 )&& ( strncasecmp(str,"POSEQ",5) == 0 ) ) { return TRAJECTORYPRIMITIVES_POSEQ; } 
             else  if ( (strLength >= 4 )&& ( strncasecmp(str,"POSE",4) == 0 ) ) { return TRAJECTORYPRIMITIVES_POSE; } 
             else  if ( (strLength >= 3 )&& ( strncasecmp(str,"POS",3) == 0 ) ) { return TRAJECTORYPRIMITIVES_POS; } 
         break; 
        }; 
     break; 
     case 'Q' : 
         if (strLength<2) { return 0; } 
         if ( strncasecmp(str,"PQ",2) == 0 ) { return TRAJECTORYPRIMITIVES_PQ; } 
     break; 
     case 'R' : 
         if (strLength<17) { return 0; } 
         if ( strncasecmp(str,"PROJECTION_MATRIX",17) == 0 ) { return TRAJECTORYPRIMITIVES_PROJECTION_MATRIX; } 
     break; 
     case 0 : /*Null Terminator*/ 
         if (strLength<1) { return 0; } 
         if ( strncasecmp(str,"P",1) == 0 ) { return TRAJECTORYPRIMITIVES_P; } 
     break; 
    }; 
 break; 
 case 'R' : 
     switch (toupper(str[1])) { 
     case 'A' : 
         if (strLength<4) { return 0; } 
         if ( strncasecmp(str,"RATE",4) == 0 ) { return TRAJECTORYPRIMITIVES_RATE; } 
     break; 
     case 'I' : 
         if (strLength<12) { return 0; } 
         if ( strncasecmp(str,"RIGID_OBJECT",12) == 0 ) { return TRAJECTORYPRIMITIVES_RIGID_OBJECT; } 
     break; 
    }; 
 break; 
 case 'S' : 
     switch (toupper(str[1])) { 
     case 'A' : 
         if (strLength<22) { return 0; } 
         if ( strncasecmp(str,"SAVED_FILE_DEPTH_SCALE",22) == 0 ) { return TRAJECTORYPRIMITIVES_SAVED_FILE_DEPTH_SCALE; } 
     break; 
     case 'C' : 
         if (strLength<11) { return 0; } 
         if ( strncasecmp(str,"SCALE_WORLD",11) == 0 ) { return TRAJECTORYPRIMITIVES_SCALE_WORLD; } 
     break; 
     case 'H' : 
         if (strLength<6) { return 0; } 
         if ( strncasecmp(str,"SHADER",6) == 0 ) { return TRAJECTORYPRIMITIVES_SHADER; } 
     break; 
     case 'I' : 
         if (strLength<6) { return 0; } 
         if ( strncasecmp(str,"SILENT",6) == 0 ) { return TRAJECTORYPRIMITIVES_SILENT; } 
     break; 
     case 'M' : 
         if (strLength<6) { return 0; } 
         if ( strncasecmp(str,"SMOOTH",6) == 0 ) { return TRAJECTORYPRIMITIVES_SMOOTH; } 
     break; 
    }; 
 break; 
 case 'T' : 
     if (strLength<9) { return 0; } 
     if ( strncasecmp(str,"TIMESTAMP",9) == 0 ) { return TRAJECTORYPRIMITIVES_TIMESTAMP; } 
 break; 
}; 
 return 0;
}
