#ifndef ACQUISITION_SETUP_H_INCLUDED
#define ACQUISITION_SETUP_H_INCLUDED

/*
     THIS IS THE SUPER HEADER FILE WHICH CONTROLS WHAT WILL BE COMPILED AND NOT !!
     IF YOU WANT TO DISABLE A FEATURE PLEASE SET IT TO 0 AND THEN ALSO COMMENT IT OUT
     BY ADDING // IN THE START OF THE LINE ,  JUST COMMENTING OUT WORKS BUT MIGHT
     CONFUSE THE GREP SCRIPT FOR COMPILING THE GRABBERS THAT USE libAcquisition.so

     THIS COULD ALSO  BE AUTOMATICALLY GENERATED...
*/

//#define BUILD_DUMMY 0 <- A disabled module should look like this :P

//You should manually set them to reflect your 3d party configuration
//if something is disabled the plugin will be compiled as a stub , it will
//basically do nothing.. :P
#define BUILD_OPENNI1 1
#define BUILD_OPENNI2 1
#define BUILD_FREENECT 1
#define BUILD_OPENGL 1
#define BUILD_NETWORK 1
#define BUILD_TEMPLATE 1 //<- This is on by default because it has no dependencies
#define BUILD_V4L2 1
//todo add more acquisition modules here..

#define PRINT_DEBUG_EACH_CALL 0

#define USE_CALIBRATION 1


#endif // ACQUISITION_SETUP_H_INCLUDED
