/*                  
This file was automatically generated @ 11-06-2015 14:05:56 using StringRecognizer                  
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
 if (strLength<3) { return 0; } 

 switch (toupper(str[0])) { 
 case 'A' : 
     switch (toupper(str[1])) { 
     case 'F' : 
         if (strLength<32) { return 0; } 
         if ( strncasecmp(str,"AFFIX_OBJ_TO_OBJ_FOR_NEXT_FRAMES",32) == 0 ) { return TRAJECTORYPRIMITIVES_AFFIX_OBJ_TO_OBJ_FOR_NEXT_FRAMES; } 
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
     if (strLength<5) { return 0; } 
     if ( strncasecmp(str,"DEBUG",5) == 0 ) { return TRAJECTORYPRIMITIVES_DEBUG; } 
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
         break; 
        }; 
     break; 
    }; 
 break; 
 case 'O' : 
     switch (toupper(str[1])) { 
     case 'B' : 
         switch (toupper(str[2])) { 
         case 'J' : 
             if ( (strLength >= 10 )&& ( strncasecmp(str,"OBJECTTYPE",10) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJECTTYPE; } 
             else  if ( (strLength >= 6 )&& ( strncasecmp(str,"OBJECT",6) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJECT; } 
             else  if ( (strLength >= 3 )&& ( strncasecmp(str,"OBJ",3) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJ; } 
             else  if ( (strLength >= 10 )&& ( strncasecmp(str,"OBJ_OFFSET",10) == 0 ) ) { return TRAJECTORYPRIMITIVES_OBJ_OFFSET; } 
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
         if (strLength<3) { return 0; } 
         if ( strncasecmp(str,"POS",3) == 0 ) { return TRAJECTORYPRIMITIVES_POS; } 
     break; 
     case 'R' : 
         if (strLength<17) { return 0; } 
         if ( strncasecmp(str,"PROJECTION_MATRIX",17) == 0 ) { return TRAJECTORYPRIMITIVES_PROJECTION_MATRIX; } 
     break; 
    }; 
 break; 
 case 'R' : 
     if (strLength<12) { return 0; } 
     if ( strncasecmp(str,"RIGID_OBJECT",12) == 0 ) { return TRAJECTORYPRIMITIVES_RIGID_OBJECT; } 
 break; 
 case 'S' : 
     switch (toupper(str[1])) { 
     case 'C' : 
         if (strLength<11) { return 0; } 
         if ( strncasecmp(str,"SCALE_WORLD",11) == 0 ) { return TRAJECTORYPRIMITIVES_SCALE_WORLD; } 
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
