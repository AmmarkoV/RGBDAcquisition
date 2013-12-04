#ifndef TOOLS_H_INCLUDED
#define TOOLS_H_INCLUDED

int XYOverRect(int x , int y , int rectx1,int recty1,int rectx2,int recty2);
int dumpCameraDepths(unsigned int moduleID , unsigned int devID , char * filename);
int dumpExtDepths(unsigned int moduleID , unsigned int devID , char * filename);

#endif // TOOLS_H_INCLUDED
