

#define EMMIT_CALIBRATION 1
#define USE_CALIBRATION 1



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
#include <iostream>
#include <unistd.h>
#include <ros/ros.h>
#include <ros/spinner.h>

#include <opencv2/opencv.hpp>
#include <opencv/cvwimage.h>
#include <opencv/highgui.h>


#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <cv_bridge/cv_bridge.h>
#include <std_srvs/Empty.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>


#include <image_transport/image_transport.h>

#include "Acquisition.h"
#include "calibration.h"

#include "rgbd_acquisition/SetScale.h"
#include <unistd.h>


#define NODE_NAME "rgbd_acquisition"




#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

//Initial Frame name , will be overwritten by launch file..!
char tfRoot[512]={"map"};

//These are the static declarations of the various parts of this ROS package
int key = 0;
double  virtual_baseline=0.0; //This is 0 and should be zero since we have a registered depth/rgb stream , however it can be changed to allow fake disparity to be generated
volatile int paused = 0;

float scaleDepth=1.0;

ros::NodeHandle * nhPtr=0;
image_transport::Publisher pubRGB;
image_transport::Publisher pubDepth;

ros::Publisher pubRGBInfo;
ros::Publisher pubDepthInfo;

int devID=0,moduleID=0;
int useRGBDAcqusition=0;
volatile static bool first = true;

int width = 640;
int height = 480;
int framerate = 30;

unsigned int draw_out = 0;
unsigned int counter = 0;

unsigned int colorWidth, colorHeight , colorChannels , colorBitsperpixel;
unsigned int depthWidth, depthHeight , depthChannels , depthBitsperpixel;


#if USE_CALIBRATION
 struct calibration calibRGB;
 struct calibration calibDepth;
#endif

enum FRAMERATE_STATES
{
    STOPPED_FRAMERATE = 0 ,
    LOW_FRAMERATE,
    MEDIUM_FRAMERATE,
    HIGH_FRAMERATE
};

unsigned int framerateState=HIGH_FRAMERATE;




unsigned int getTicks()
{
 return cvGetTickCount();
}

void switchDrawOutTo(unsigned int newVal)
{
   if ( (draw_out==1) && (newVal==0) )
   {
  	 cv::destroyWindow("Depth");
   	 cv::destroyWindow("RGB");

   	 cv::destroyAllWindows();
     cv::waitKey(1);

	 cv::waitKey(1);
   }
   draw_out = newVal;
}



//----------------------------------------------------------
//Advertised Service switches

bool setStoppedFramerate(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    framerateState=STOPPED_FRAMERATE;
    return true;
}

bool setLowFramerate(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    framerateState=LOW_FRAMERATE;
    return true;
}

bool setMediumFramerate(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    framerateState=MEDIUM_FRAMERATE;
    return true;
}

bool setHighFramerate(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    framerateState=HIGH_FRAMERATE;
    return true;
}



bool saveROSCalibrationFile(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
  struct calibration calib;
  acquisitionGetColorCalibration(moduleID,devID,&calib);

    char cwd[1024];
   if (getcwd(cwd, sizeof(cwd)) != NULL)
       fprintf(stdout, "Current working dir: %s\n", cwd);
   else
    {   
      perror("getcwd() error");
      return true;
    }
  WriteCalibrationROS("calibration.yaml",&calib);
  return true;
}


bool visualizeOn(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    switchDrawOutTo(1);
    return true;
}

bool visualizeOff(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    switchDrawOutTo(0);
    return true;
}


bool terminate(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    ROS_INFO("Stopping RGBDAcquisition");
    exit(0);
    return true;
}


bool pause(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    return true;
}

bool resume(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    return true;
}



bool setScale( rgbd_acquisition::SetScale::Request  &request,
                  rgbd_acquisition::SetScale::Response &response )
{
   scaleDepth =  request.factor;
   fprintf(stderr,"depth scale is now %0.2f \n",scaleDepth);

   if ( scaleDepth==1.0 ) { fprintf(stderr,"scaling is now deactivated\n"); }
   response.ok=1;

   return true ;
}





int doDrawOutFrame( unsigned char * rgbFrame , unsigned int rgbWidth , unsigned int rgbHeight ,
                     unsigned short * depthFrame , unsigned int depthWidth , unsigned int depthHeight )
{
  if (!draw_out)  { return 0; }
  //-----------------------------------------------------------------------------------------------------------------------
  cv::Mat wrappedColor(rgbHeight,rgbWidth, CV_8UC3,  rgbFrame , rgbWidth*3); // does not copy
  cv::Mat rgbTmp = wrappedColor.clone();
  //Take care of drawing stuff as visual output
  cv::Mat bgrMat,rgbMat(rgbHeight,rgbWidth,CV_8UC3,rgbTmp.data,3*rgbWidth);
  cv::cvtColor(rgbMat,bgrMat, CV_RGB2BGR);// opencv expects the image in BGR format


  //-----------------------------------------------------------------------------------------------------------------------
  cv::Mat wrappedDepth(depthHeight,depthWidth, CV_16UC1 ,depthFrame, depthWidth*2); // does not copy

  cv::Mat depthNorm;
  cv::normalize(wrappedDepth,depthNorm,0,255,CV_MINMAX,CV_8UC1);

  //After we have our bgr Frame ready and we added the FPS text , lets show it!
  cv::imshow("Depth",depthNorm);
  cv::imshow("RGB",bgrMat);

  return 1;
}


int doDrawOut(  )
{

   return doDrawOutFrame(acquisitionGetColorFrame(moduleID,devID),colorWidth,colorHeight,
                          acquisitionGetDepthFrame(moduleID,devID),depthWidth,depthHeight);
}


int getKeyPressed()
{
   if  (draw_out==1) { return cv::waitKey(3)&0x000000FF; }
   return ' '; /*No Windows -> No keys :P */
}



bool publishImagesFrames(unsigned char * color , unsigned int colorWidth , unsigned int colorHeight ,
                          unsigned short * depth , unsigned int depthWidth , unsigned int depthHeight ,
                           struct calibration  * calib
                         )
{
  //We want to populate this
  ros::Time sampleTime = ros::Time::now();
  cv_bridge::CvImage out_RGB_msg;
   out_RGB_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
   out_RGB_msg.header.frame_id= tfRoot;
   out_RGB_msg.header.stamp= sampleTime;

  cv_bridge::CvImage out_Depth_msg;
   out_Depth_msg.header.frame_id= tfRoot;
   out_Depth_msg.header.stamp= sampleTime;
   out_Depth_msg.encoding = sensor_msgs::image_encodings::TYPE_16UC1; // Or whatever


  //convert & publish RGB Stream - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //IplImage *imageRGB = cvCreateImageHeader( cvSize(colorWidth,colorHeight), IPL_DEPTH_8U ,3);
  //imageRGB->imageData = (char *) color;
  //out_RGB_msg.header   = in_msg->header; // Same timestamp and tf frame as input image

    cv::Mat imageRGBMat(cv::Size(colorWidth, colorHeight), CV_8UC3, (char *) color, cv::Mat::AUTO_STEP);
    out_RGB_msg.image = imageRGBMat; // Your cv::Mat
    pubRGB.publish(out_RGB_msg.toImageMsg());


  //convert & publish Depth Stream - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //IplImage *imageDepth = cvCreateImageHeader( cvSize(depthWidth,depthHeight), IPL_DEPTH_16U ,1);
  //imageDepth->imageData = (char *) depth;
  //out_Depth_msg.image    = imageDepth; // Your cv::Mat

   cv::Mat imageDepthMat(cv::Size(depthWidth, depthHeight), CV_16UC1, (char *) depth , cv::Mat::AUTO_STEP);
   out_Depth_msg.image    = imageDepthMat;
   pubDepth.publish(out_Depth_msg.toImageMsg());


   //---------------------------------------------------------------------------------------------------
   //  CAMERA INFORMATION BROADCAST --------------------------------------------------------------------
   //---------------------------------------------------------------------------------------------------
   #if EMMIT_CALIBRATION
   if (calib!=0)
   {
    int i=0;

    sensor_msgs::CameraInfo cal;
    cal.header.stamp = sampleTime;
    cal.width=colorWidth; cal.height=colorHeight;

    cal.D.resize(5, 0.0);
    cal.distortion_model=sensor_msgs::distortion_models::PLUMB_BOB;
    cal.D[0]=calib->k1; cal.D[1]=calib->k2; cal.D[2]=calib->p1; cal.D[3]=calib->p2; cal.D[4]=calib->k3;

    for (i=0; i<9; i++) { cal.K[i]=calib->intrinsic[i]; }

    cal.R[0]=1.0; cal.R[1]=0.0; cal.R[2]=0.0;
    cal.R[3]=0.0; cal.R[4]=1.0; cal.R[5]=0.0;
    cal.R[6]=0.0; cal.R[7]=0.0; cal.R[8]=1.0;

    for (i=0; i<12; i++) { cal.P[i]=0.0; }

    //Base line for fake disparity !
    //P[3] should be 0.0 since we have a registered feed but we add a baseline for fake disparity here
    //Please note that the virtual baseline is multiplied by -1 and fX to fit the "ROS" way of
    //declaring a baseline as seen here https://github.com/ros-perception/image_pipeline/blob/hydro-devel/depth_image_proc/src/nodelets/disparity.cpp#L136
    //for the record , I added an issue https://github.com/ros-perception/image_pipeline/issues/58 but it seems this is the way to do it..
     cal.P[3]=-1.0 * virtual_baseline * calib->intrinsic[CALIB_INTR_FX];
    // ----------------------------------------------------------------------

    cal.P[0]=calib->intrinsic[CALIB_INTR_FX];
    cal.P[5]=calib->intrinsic[CALIB_INTR_FY];
    cal.P[2]=calib->intrinsic[CALIB_INTR_CX];
    cal.P[6]=calib->intrinsic[CALIB_INTR_CY];
    cal.P[10]=1.0;

    pubRGBInfo.publish(cal);
    cal.width=depthWidth; cal.height=depthHeight;
    pubDepthInfo.publish(cal);
   }
   #endif
   //---------------------------------------------------------------------------------------------------
   //---------------------------------------------------------------------------------------------------
   //---------------------------------------------------------------------------------------------------


   //Spin ROS one time
   ros::spinOnce();

   //Deallocate constructed OpenCV images
   //cvReleaseImageHeader( &imageRGB );
   //cvReleaseImageHeader( &imageDepth );

   return true;
}

int scaleDepthFrame(unsigned short * depthFrame , unsigned int width , unsigned int height , float scalingFactor)
{
  if (depthFrame==0) { return 0; }

  unsigned short * depthFramePtr = depthFrame;
  unsigned short * depthFrameLimit =  depthFrame + width * height; //Is a *2 needed here ( dont think so :P )


  float curValue=0.0;
  while (depthFramePtr<depthFrameLimit)
  {
    curValue=(float) *depthFramePtr;
    curValue=curValue * scalingFactor;
    *depthFramePtr = (unsigned short) curValue;

    ++depthFramePtr;
  }
 return 1;
}


int publishImages()
{
   struct calibration rgbCalibration;
   acquisitionGetColorCalibration(moduleID,devID,&rgbCalibration);

   unsigned short * depthOriginal = acquisitionGetDepthFrame(moduleID,devID);
   unsigned short * depthUsed = depthOriginal;
   if ( scaleDepth!=1.0)
   {
     depthUsed = (unsigned short* ) malloc( colorWidth * colorHeight * sizeof(unsigned short) * 1 );
     if (depthUsed==0)
     {
       fprintf(stderr,"Failed to allocate scaled Depth , falling back to original\n");
       depthUsed=depthOriginal;
     } else
     {
         memcpy(depthUsed,depthOriginal, depthWidth * depthHeight * sizeof(unsigned short) * 1);
         if (!scaleDepthFrame(depthUsed, depthWidth , depthHeight , scaleDepth) )
         {
           fprintf(stderr,"Failed to scale depth..\n");
         }
     }
   }


   int retres = publishImagesFrames(acquisitionGetColorFrame(moduleID,devID), colorWidth , colorHeight ,
                                    depthUsed, depthWidth , depthHeight , &rgbCalibration );


   if (depthUsed!=depthOriginal)
   {
     free(depthUsed);
   }

   return retres;
}


//----------------------------------------------------------


void loopEvent()
{
  //ROS_INFO("Loop Event started");
  //We spin from this thread to keep thread synchronization problems at bay
  ros::spinOnce();

  acquisitionSnapFrames(moduleID,devID);
  acquisitionGetColorFrameDimensions(moduleID , devID , &colorWidth, &colorHeight , &colorChannels , &colorBitsperpixel);
  acquisitionGetDepthFrameDimensions(moduleID , devID , &depthWidth, &depthHeight , &depthChannels , &depthBitsperpixel);


  if (framerateState!=STOPPED_FRAMERATE)
   {
      publishImages();
   }

   // if we got a depth and rgb frames , lets go
   key=getKeyPressed(); //Post our geometry to ROS
   //If we draw out we have to visualize the hand pose , set up windows , put out text etc.

    doDrawOut();  // <- end of we have depth and rgb code
}


int main(int argc, char **argv)
{
   ROS_INFO("Starting Up!!");

   int i=system("./downloadDependencies.sh");
   if (i!=0 ) { ROS_INFO("Could not check for missing dependencies of rgbd_acquisition"); }

   try
	{
	 ROS_INFO("Initializing ROS");
  	 ros::init(argc, argv, NODE_NAME);
     ros::start();

     ros::NodeHandle nh;
     nhPtr = &nh;

     std::string from;
     std::string module;
     std::string name;
     std::string camera;
     std::string frame; 
     int highRate,midRate,lowRate;

     ros::NodeHandle private_node_handle_("~");
     private_node_handle_.param("name", name, std::string("rgbd_acquisition"));
     private_node_handle_.param("camera", camera, std::string("camera"));
     private_node_handle_.param("frame", frame, std::string("frame"));

     private_node_handle_.param("width", width, int(640));
     private_node_handle_.param("height", height, int(480));
     private_node_handle_.param("framerate", framerate , int(30)); 
     private_node_handle_.param("highRate", highRate, int(30));
     private_node_handle_.param("midRate", midRate, int(15));
     private_node_handle_.param("lowRate", lowRate, int(5));
     highRate=framerate;
     //private_node_handle_.param("scaleDepth", scaleDepth, float(1.0));

     private_node_handle_.param("deviceID", from, std::string(""));
     private_node_handle_.param("moduleID", module, std::string(""));
     private_node_handle_.param("virtual_baseline", virtual_baseline, 0.0);

     //Pass root frame for TF poses
     strcpy(tfRoot,frame.c_str());


     //Decide on devID
/*
     if (from.length()==0) { } else
     if (from.length()<=2) { devID=atoi( from.c_str() ); from.clear(); ROS_INFO("Using OpenNI2 Serializer to get device"); }*/



     std::cout<<"RGBDAcquisition Starting settings ----------------"<<std::endl;

     char cwd[1024];
     if (getcwd(cwd, sizeof(cwd)) != NULL)
      std::cout<<"Current working dir : " << cwd << std::endl;

     std::cout<<"Name : "<<name<<std::endl;
     std::cout<<"Camera : "<<camera<<std::endl;
     std::cout<<"Frame : "<<frame<<std::endl;
     std::cout<<"TF Root : "<<tfRoot<<std::endl;
     std::cout<<"Mode : "<<width<<"x"<<height<<":"<<framerate<<std::endl;
     std::cout<<"High Rate : "<<highRate<<std::endl;
     std::cout<<"Mid Rate : "<<midRate<<std::endl;
     std::cout<<"Low Rate : "<<lowRate<<std::endl;
     std::cout<<"virtual_baseline : "<<virtual_baseline<<std::endl;
     std::cout<<"Module_id : "<<module<<" length "<<module.length()<<"  devID : "<<moduleID<<std::endl;
     std::cout<<"Device_id : "<<from<<" length "<<from.length()<<"  devID : "<<devID<<std::endl;
     std::cout<<"--------------------------------------------------"<<std::endl;


     ros::Rate high_loop_rate(highRate); //  hz should be our target performance
     ros::Rate medium_loop_rate(midRate); //  hz should be our target performance
     ros::Rate low_loop_rate(lowRate); //  hz should be our target performance
     framerateState=HIGH_FRAMERATE; //By default we go to medium!

     //We advertise the services we want accessible using "rosservice call *w/e*"
     ros::ServiceServer saveROSCalibrationService  = nh.advertiseService(name+"/saveROSCalibration", saveROSCalibrationFile);
     ros::ServiceServer visualizeOnService         = nh.advertiseService(name+"/visualize_on", visualizeOn);
     ros::ServiceServer visualizeOffService        = nh.advertiseService(name+"/visualize_off", visualizeOff);
     ros::ServiceServer terminateService           = nh.advertiseService(name+"/terminate", terminate);
     ros::ServiceServer pauseService               = nh.advertiseService(name+"/pause", pause);
     ros::ServiceServer resumeService              = nh.advertiseService(name+"/resume", resume);
     ros::ServiceServer setScaleService            = nh.advertiseService(name+"/setScale", setScale);

     //Framerate switches
     ros::ServiceServer setStoppedFramerateService   = nh.advertiseService(name+"/setStoppedFramerate", setStoppedFramerate);
     ros::ServiceServer setLowFramerateService       = nh.advertiseService(name+"/setLowFramerate", setLowFramerate);
     ros::ServiceServer setMediumFramerateService    = nh.advertiseService(name+"/setMediumFramerate", setMediumFramerate);
     ros::ServiceServer setNormalFramerateService    = nh.advertiseService(name+"/setNormalFramerate", setMediumFramerate);
     ros::ServiceServer setHighFramerateService      = nh.advertiseService(name+"/setHighFramerate", setHighFramerate);


     //Output RGB Image
     image_transport::ImageTransport it(nh);

     pubRGB = it.advertise(camera+"/rgb/image_rect_color", 1);
     pubRGBInfo = nh.advertise<sensor_msgs::CameraInfo>(camera+"/rgb/camera_info",1);
     std::cout<<"ROS RGB Topic name : "<<camera<<"/rgb/image_rect_color"<<std::endl;

     pubDepth = it.advertise(camera+"/depth_registered/image_rect", 1);
     pubDepthInfo = nh.advertise<sensor_msgs::CameraInfo>(camera+"/depth_registered/camera_info",1);
     std::cout<<"ROS Depth Topic name : "<<camera<<"/depth_registered/image_rect"<<std::endl;

      //---------------------------------------------------------------------------------------------------
      //This code segment waits for a valid first frame to come and initialize the focal lengths etc..
      //If there is no first frame it times out after a little while displaying a relevant error message
      //---------------------------------------------------------------------------------------------------

  moduleID = getModuleIdFromModuleName((char*) module.c_str());

  if (!acquisitionIsModuleAvailiable(moduleID))
   {
       fprintf(stderr,RED "\n\n\nThe module you are trying to use is not linked in this build\n of the Acquisition library..\n\n\n" NORMAL);
       return 1;
   }

  if (!acquisitionStartModule(moduleID,16 /*maxDevices*/ , 0 ))
   {
       fprintf(stderr,RED "Could not start module %s ..\n" NORMAL,getModuleNameFromModuleID(moduleID));
       return 1;
   }


   #if EMMIT_CALIBRATION
    acquisitionSetColorCalibration(moduleID,devID,&calibRGB);
    acquisitionSetDepthCalibration(moduleID,devID,&calibDepth);
   #endif // EMMIT_CALIBRATION

  acquisitionMapDepthToRGB(moduleID,devID);

  std::cout<<"Trying to open capture device.."<<std::endl;
  ROS_INFO("Trying to open capture device..");
   if (!acquisitionOpenDevice(moduleID,devID,(char* ) from.c_str(),width,height,framerate))
        {
          fprintf(stderr,RED "Could not open device %u ( %s ) of module %s  ..\n" NORMAL,devID,from.c_str(),getModuleNameFromModuleID(moduleID));
          return 1;
        }


	 ROS_INFO("Done Initializing RGBDAcqusition , now entering grab loop..");
	  while ( ( key!='q' ) && (ros::ok()) )
		{
                  loopEvent(); //<- this keeps our ros node messages handled up until synergies take control of the main thread

                  switch (framerateState)
                  {
                    case HIGH_FRAMERATE   :   high_loop_rate.sleep();    break;
                    case MEDIUM_FRAMERATE :   medium_loop_rate.sleep();  break;
                    case LOW_FRAMERATE    :   low_loop_rate.sleep();     break;
                    default :
                           low_loop_rate.sleep();
                    break;
                  }


		}

      switchDrawOutTo(0);

      acquisitionCloseDevice(moduleID,devID);
      acquisitionStopModule(moduleID);
	}
	catch(std::exception &e) { ROS_ERROR("Exception: %s", e.what()); return 1; }
	catch(...)               { ROS_ERROR("Unknown Error"); return 1; }
	ROS_ERROR("Shutdown complete");
	return 0;
}
