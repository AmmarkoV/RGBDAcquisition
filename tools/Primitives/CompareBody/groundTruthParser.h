#ifndef GROUNDTRUTHPARSER_H_INCLUDED
#define GROUNDTRUTHPARSER_H_INCLUDED


extern "C"
{


int groundTruthProviderStart(unsigned int moduleID , unsigned int devID , const char * directory);
int groundTruthProviderStop();

const char * getGroundTruthPath();

int groundTruthProviderSetupSkinnedModel(
                                    char * modelPath,
                                    double * minimumValues,
                                    double * varianceValues,
                                    double * maximumValues,
                                    unsigned int numberOfValues
                                   );


int groundTruthProviderNewFrame(
                            unsigned char * colorFrame , unsigned int colorWidth , unsigned int colorHeight ,
                            unsigned short * depthFrame  , unsigned int depthWidth , unsigned int depthHeight ,
                            struct calibration  * fc ,
                            unsigned int frameNumber
                          );



float *  getGroundTruth(unsigned char * colorFrame , unsigned int colorWidth , unsigned int colorHeight ,
                        unsigned short * depthFrame  , unsigned int depthWidth , unsigned int depthHeight ,
                        struct calibration  * fc ,
                        unsigned int frameNumber ,
                        unsigned int *groundTruthLength
                        );



}

#endif

