#ifndef DEPTHCLASSIFIER_H_INCLUDED
#define DEPTHCLASSIFIER_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

struct depthClassifierPoint
{
  unsigned int x,y,depth,minAccepted,maxAccepted,samples;
};


struct depthClassifier
{
  unsigned int patchWidth , patchHeight;

  unsigned int maxPointList;
  unsigned int currentPointList;

  unsigned int depthBase;
  unsigned int totalSamples;

  struct depthClassifierPoint pointList[64];
};



int trainDepthClassifier(struct depthClassifier * dc ,
                         unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                         unsigned int tileWidth , unsigned int tileHeight);

unsigned int compareDepthClassifier(struct depthClassifier * dc ,
                                    unsigned short * target,  unsigned int tX,  unsigned int tY  , unsigned int targetWidth , unsigned int targetHeight ,
                                    unsigned int tileWidth , unsigned int tileHeight);

int printDepthClassifier(char * filename , struct depthClassifier * dc );


#ifdef __cplusplus
}
#endif


#endif // DEPTHCLASSIFIER_H_INCLUDED
