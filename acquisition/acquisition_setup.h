#ifndef ACQUISITION_SETUP_H_INCLUDED
#define ACQUISITION_SETUP_H_INCLUDED

/*
     THIS IS THE SUPER HEADER FILE WHICH CONTROLS WHAT WILL BE COMPILED AND NOT !!
     IF YOU WANT TO DISABLE A FEATURE PLEASE SET IT TO 0 AND THEN ALSO COMMENT IT OUT
     BY ADDING // IN THE START OF THE LINE ,  JUST COMMENTING OUT WORKS BUT MIGHT
     CONFUSE THE GREP SCRIPT FOR COMPILING THE GRABBERS THAT USE libAcquisition.so

     THIS COULD ALSO  BE AUTOMATICALLY GENERATED...
*/

//#define USE_DUMMY 0 <- A disabled module should look like this :P

//You should manually set them to reflect your 3d party configuration
#define USE_OPENNI1 0
#define USE_OPENNI2 0
#define USE_FREENECT 0
#define USE_OPENGL 0
#define USE_TEMPLATE 1
#define USE_V4L2 0

//todo add more acquisition modules here..

#endif // ACQUISITION_SETUP_H_INCLUDED
