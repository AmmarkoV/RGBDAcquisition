#ifndef ACQUISITION_H_INCLUDED
#define ACQUISITION_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

enum Acquisition_Possible_Modules
{
    NO_ACQUISITION_MODULE = 0,
    V4L2_ACQUISITION_MODULE ,
    V4L2STEREO_ACQUISITION_MODULE ,
    FREENECT_ACQUISITION_MODULE ,
    OPENNI1_ACQUISITION_MODULE  ,
    OPENNI2_ACQUISITION_MODULE  ,
    OPENGL_ACQUISITION_MODULE  ,
    TEMPLATE_ACQUISITION_MODULE  ,
   //--------------------------------------------
    NUMBER_OF_POSSIBLE_MODULES
};

typedef unsigned int ModuleIdentifier;
typedef unsigned int DeviceIdentifier;


struct calibration
{
  /* CAMERA INTRINSIC PARAMETERS */
  char intrinsicParametersSet;
  float intrinsic[9];
  float k1,k2,p1,p2,k3;

  /* CAMERA EXTRINSIC PARAMETERS */
  char extrinsicParametersSet;
  float extrinsicRotationRodriguez[3];
  float extrinsicTranslation[3];
};

int acquisitionSimulateTime(unsigned long timeInMillisecs);

int saveRawImageToFile(char * filename,char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);

char * convertShortDepthToRGBDepth(short * depth,unsigned int width , unsigned int height);
char * convertShortDepthToCharDepth(short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);

ModuleIdentifier getModuleIdFromModuleName(char * moduleName);
int acquisitionGetModulesCount();
char * getModuleStringName(ModuleIdentifier moduleID);
int acquisitionIsModuleLinked(ModuleIdentifier moduleID);

int acquisitionStartModule(ModuleIdentifier moduleID,unsigned int maxDevices,char * settings);
int acquisitionStopModule(ModuleIdentifier moduleID);
int acquisitionGetModuleDevices(ModuleIdentifier moduleID);
int acquisitionOpenDevice(ModuleIdentifier moduleID,DeviceIdentifier devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);
int acquisitionCloseDevice(ModuleIdentifier moduleID,DeviceIdentifier devID);

 int acquisitionSeekFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int seekFrame);
int acquisitionSnapFrames(ModuleIdentifier moduleID,DeviceIdentifier devID);


int acquisitionGetColorCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib);
int acquisitionGetDepthCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib);
int acquisitionSetColorCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib);
int acquisitionSetDepthCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib);

unsigned long acquisitionGetColorTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID);
unsigned long acquisitionGetDepthTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID);


char * acquisitionGetColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID);
unsigned int acquisitionCopyColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * mem,unsigned int memlength);
unsigned int acquisitionCopyColorFramePPM(ModuleIdentifier moduleID,DeviceIdentifier devID,char * mem,unsigned int memlength);

short * acquisitionGetDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID);
unsigned int acquisitionCopyDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,short * mem,unsigned int memlength);
unsigned int acquisitionCopyDepthFramePPM(ModuleIdentifier moduleID,DeviceIdentifier devID,short * mem,unsigned int memlength);

int acquisitionGetDepth3DPointAtXY(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d , float *x, float *y , float *z  );
int acquisitionGetColorFrameDimensions(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int * width , unsigned int * height , unsigned int * channels , unsigned int * bitsperpixel );
int acquisitionGetDepthFrameDimensions(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int * width , unsigned int * height , unsigned int * channels , unsigned int * bitsperpixel );



int acquisitionSavePCDPointCoud(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);
int acquisitionSaveColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);
int acquisitionSaveDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);
int acquisitionSaveDepthFrame1C(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);
 int acquisitionSaveColoredDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);

double acqusitionGetColorFocalLength(ModuleIdentifier moduleID,DeviceIdentifier devID);
double acqusitionGetColorPixelSize(ModuleIdentifier moduleID,DeviceIdentifier devID);

double acqusitionGetDepthFocalLength(ModuleIdentifier moduleID,DeviceIdentifier devID);
double acqusitionGetDepthPixelSize(ModuleIdentifier moduleID,DeviceIdentifier devID);

int acquisitionMapDepthToRGB(ModuleIdentifier moduleID,DeviceIdentifier devID);

#ifdef __cplusplus
}
#endif

#endif // ACQUISITION_H_INCLUDED
