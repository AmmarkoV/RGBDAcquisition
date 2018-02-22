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

#include <unistd.h>


#define NODE_NAME "ros_dumper"




#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

#define MAX_RECORDED_FRAMES 10000
#define USE_NONDEFAULT_CALIBRATIONS 1

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

float depthScale=1.0;
int useFloatDepth=0;

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

unsigned int recording=1;
unsigned int recordedFrames=0;
unsigned int frameTimestamp=0;

ros::Time begin;

unsigned int doVisualization=1;
int rate=30;

#if USE_NONDEFAULT_CALIBRATIONS
 typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> RgbdSyncPolicy;
#else
 typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> RgbdSyncPolicy;
#endif

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
//----------------------------------------------------------



unsigned short * get16UC1from32FC1(float * f , unsigned int width ,unsigned int height , float scale )
{
 float * fp = f;
 float * fpLimit = f + (width * height);

 unsigned short * out = (unsigned short * ) malloc(sizeof(unsigned short) * width * height);
 unsigned short * outP = out;
 if (out!=0)
 {
  if (scale==1.0)
   {
     //Fastpath no scaling
     while ( fp < fpLimit )
      {
       *outP = (unsigned short) *fp;
       ++outP;
       ++fp;
      }
   } else
   {
     while ( fp < fpLimit )
      {
       *outP = (unsigned short) (*fp * scale);
       ++outP;
       ++fp;
      }
   }
 }

 return out;
}


int doDrawOutFrame( unsigned char * rgbFrame , unsigned int rgbWidth , unsigned int rgbHeight ,
                     unsigned short * depthFrame , unsigned int depthWidth , unsigned int depthHeight )
{
  if (!doVisualization) {return 0;}
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
  cv::imshow("Received Depth",depthNorm);
  cv::imshow("Received RGB",bgrMat);
  cv::waitKey(1);

  return 1;
}


int appendTimestamps(const char * fileout,unsigned int recordedFrames,unsigned long timestamp)
{
  if (recordedFrames==0)
  {
     FILE * fp = fopen(fileout,"w");
     if (fp!=0) {fprintf(fp,"%%time\n%lu\n",timestamp); fclose(fp); return 1; }
  }
   else
  {
    FILE * fp = fopen(fileout,"a");
    if (fp!=0) { fprintf(fp,"%lu\n",timestamp); fclose(fp); return 1; }
  }
  return 0;
}


#if USE_NONDEFAULT_CALIBRATIONS
//RGBd Callback is called every time we get a new pair of frames , it is synchronized to the main thread
void rgbdCallback(const sensor_msgs::Image::ConstPtr rgb_img_msg,
                    const sensor_msgs::Image::ConstPtr depth_img_msg,
                     const sensor_msgs::CameraInfo::ConstPtr camera_info_msg )
{
 if (paused) { return; } //If we are paused spend no time with new input
 //Using Intrinsic camera matrix for the raw (distorted) input images.

 colorWidth = rgb_img_msg->width;   colorHeight = rgb_img_msg->height;
 depthWidth = depth_img_msg->width; depthHeight = depth_img_msg->height;
 int i=0;
 //calib.width = colorWidth; calib.height = colorHeight;
 //for (i=0; i<9; i++) { calib.intrinsic[i]=camera_info_msg->K[i]; }
 //This hangs -> for (i=0; i<5; i++) { calib.intrinsic[i]=camera_info_msg->D[i]; }
 //TODO maybe populate calib.extrinsics here

  //A new pair of frames has arrived , copy and convert them so that they are ready
 cv_bridge::CvImageConstPtr orig_rgb_img;
 cv_bridge::CvImageConstPtr orig_depth_img;
 orig_rgb_img = cv_bridge::toCvCopy(rgb_img_msg, "rgb8");

 unsigned short* depthPTR = 0;
 int usedTemporaryBufferForConversion = 0;

 if (useFloatDepth)
 {
  orig_depth_img = cv_bridge::toCvCopy(depth_img_msg, sensor_msgs::image_encodings::TYPE_32FC1);
  depthPTR = get16UC1from32FC1( (float*) orig_depth_img->image.data , depthWidth , depthHeight , depthScale);
  usedTemporaryBufferForConversion = 1;
 } else
 {
  orig_depth_img = cv_bridge::toCvCopy(depth_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  depthPTR = (unsigned short*) orig_depth_img->image.data;
 }


 if (recording)
    {
       ros::Time time = ros::Time::now();
       if ( recordedFrames==0) { begin=time; }


       ros::Duration timestamp = time - begin;
       //appendTimestamps("imageTimestamps",recordedFrames,(unsigned long) timestamp.toNSec()/1000000);
       appendTimestamps("imageTimestamps",recordedFrames,time.toNSec());

       char filenameOut[512];
       snprintf(filenameOut,512,"colorFrame_0_%05u.jpg",recordedFrames);
       cv::imwrite(filenameOut,orig_rgb_img->image);

       snprintf(filenameOut,512,"depthFrame_0_%05u.png",recordedFrames);
       cv::imwrite(filenameOut,orig_depth_img->image);

      ++recordedFrames;
    }
          if (recordedFrames>MAX_RECORDED_FRAMES)
          {
            fprintf(stderr,"Automatic Cut Off of recording activated..");
           // stopDumpInternal();
          }

/*
   runServicesThatNeedColorAndDepth((unsigned char*) orig_rgb_img->image.data, colorWidth , colorHeight ,
                                    (unsigned short*) depthPTR ,  depthWidth , depthHeight ,
                                     &calib , frameTimestamp );*/


//if (useFloatDepth)
{
  doDrawOutFrame((unsigned char*) orig_rgb_img->image.data, colorWidth , colorHeight ,
                                   (unsigned short*) depthPTR ,  depthWidth , depthHeight);
}

 if (usedTemporaryBufferForConversion) { free(depthPTR); }
 ++frameTimestamp;

 //After running (at least) once it is not a first run any more!
 first = false;
 return;
}
#else
//RGBd Callback is called every time we get a new pair of frames , it is synchronized to the main thread
void rgbdCallbackNoCalibration(const sensor_msgs::Image::ConstPtr rgb_img_msg,
                                 const sensor_msgs::Image::ConstPtr depth_img_msg  )
{
if (paused) { return; } //If we are paused spend no time with new input
 //Using Intrinsic camera matrix for the raw (distorted) input images.

 colorWidth = rgb_img_msg->width;   colorHeight = rgb_img_msg->height;
 depthWidth = depth_img_msg->width; depthHeight = depth_img_msg->height;
 int i=0;

  //A new pair of frames has arrived , copy and convert them so that they are ready
 cv_bridge::CvImageConstPtr orig_rgb_img;
 cv_bridge::CvImageConstPtr orig_depth_img;
 orig_rgb_img = cv_bridge::toCvCopy(rgb_img_msg, "rgb8");

 unsigned short* depthPTR = 0;
 int usedTemporaryBufferForConversion = 0;

 if (useFloatDepth)
 {
  orig_depth_img = cv_bridge::toCvCopy(depth_img_msg, sensor_msgs::image_encodings::TYPE_32FC1);
  depthPTR = get16UC1from32FC1( (float*) orig_depth_img->image.data , depthWidth , depthHeight , depthScale);
  usedTemporaryBufferForConversion = 1;
 } else
 {
  orig_depth_img = cv_bridge::toCvCopy(depth_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  depthPTR = (unsigned short*) orig_depth_img->image.data;
 }


 if (recording)
    {
       ros::Time time = ros::Time::now();
       if ( recordedFrames==0) { begin=time; }


       ros::Duration timestamp = time - begin;
       //appendTimestamps("imageTimestamps",recordedFrames,(unsigned long) timestamp.toNSec()/1000000);
       appendTimestamps("imageTimestamps",recordedFrames,time.toNSec());

       char filenameOut[512];
       snprintf(filenameOut,512,"colorFrame_0_%05u.jpg",recordedFrames);
       cv::imwrite(filenameOut,orig_rgb_img->image);

       snprintf(filenameOut,512,"depthFrame_0_%05u.png",recordedFrames);
       cv::imwrite(filenameOut,orig_depth_img->image);

      ++recordedFrames;
    }
          if (recordedFrames>MAX_RECORDED_FRAMES)
          {
            fprintf(stderr,"Automatic Cut Off of recording activated..");
           // stopDumpInternal();
          }

/*
   runServicesThatNeedColorAndDepth((unsigned char*) orig_rgb_img->image.data, colorWidth , colorHeight ,
                                    (unsigned short*) depthPTR ,  depthWidth , depthHeight ,
                                     &calib , frameTimestamp );*/


//if (useFloatDepth)
{
  doDrawOutFrame((unsigned char*) orig_rgb_img->image.data, colorWidth , colorHeight ,
                                   (unsigned short*) depthPTR ,  depthWidth , depthHeight);
}

 if (usedTemporaryBufferForConversion) { free(depthPTR); }
 ++frameTimestamp;

 //After running (at least) once it is not a first run any more!
 first = false;
 return;
}
#endif

void loopEvent()
{
  //ROS_INFO("Loop Event started");
  //We spin from this thread to keep thread synchronization problems at bay
  ros::spinOnce();
     ros::Rate loop_rate_fullSpeed(30); //  hz should be our target performance
  loop_rate_fullSpeed.sleep();
}


int main(int argc, char **argv)
{
   ROS_INFO("Starting Up!!");
   try
	{
	 ROS_INFO("Initializing ROS");
  	 ros::init(argc, argv, NODE_NAME);
     ros::start();

     ros::NodeHandle nh;
     ros::NodeHandle private_node_handle_("~");

     std::string name;
     std::string HTTPTarget;
     std::string fromDepthTopic;
     std::string fromDepthTopicInfo;
     std::string fromRGBTopic;
     std::string fromRGBTopicInfo;
     std::string frame;

     private_node_handle_.param("frame", frame, std::string("frame"));
     private_node_handle_.param("HTTPTarget", HTTPTarget, std::string("127.0.0.1"));
     private_node_handle_.param("fromDepthTopic", fromDepthTopic, std::string("/camera/depth_registered/image_raw"));
     private_node_handle_.param("fromDepthTopicInfo", fromDepthTopicInfo, std::string("/camera/depth_registered/camera_info"));
     private_node_handle_.param("fromRGBTopic", fromRGBTopic, std::string("/camera/rgb/image_raw"));
     private_node_handle_.param("fromRGBTopicInfo", fromRGBTopicInfo, std::string("/camera/rgb/camera_info"));
     private_node_handle_.param("name", name, std::string("skeleton_detector"));
     private_node_handle_.param("rate", rate, int(5));
     private_node_handle_.param("depth32FC1", useFloatDepth, int(0));
     private_node_handle_.param("depthScale", depthScale, float(1.0));

     ros::Rate loop_rate_ultra_low(2); //  hz should be our target performance
     ros::Rate loop_rate(rate); //  hz should be our target performance
     unsigned int fastRate = rate*3;
     if (fastRate > 30) { fastRate=30; } //Cap fast speed at 30Hz ( i.e. frame rate of Depth camera )
     ros::Rate loop_rate_fast(fastRate); //  hz should be our target performance
     ros::Rate loop_rate_fullSpeed(30); //  hz should be our target performance

     //Pass root frame for TF poses
     strcpy(tfRoot,frame.c_str());

     //We advertise the services we want accessible using "rosservice call *w/e*"


     //ros::ServiceServer setPitchService         = nh.advertiseService(name+"/setPitch", setPitch);
     //ros::ServiceServer setHTTPTargetService      = nh.advertiseService(name+"/setHTTPTarget", httpTarget);

     //Make our rostopic cmaera grabber
     message_filters::Synchronizer<RgbdSyncPolicy> *sync;

     std::cerr<<"\n\n\nros_dump , RGB feed "<<fromRGBTopic<<" \n";
     std::cerr<<"ros_dump , RGB Info "<<fromRGBTopicInfo<<" \n";
     std::cerr<<"ros_dump , Depth feed "<<fromDepthTopic<<" \n";
     std::cerr<<"ros_dump , Depth Info "<<fromDepthTopicInfo<<" \n";
     std::cerr<<"ros_dump , TF Parent ("<<frame<<") \n";


    image_transport::ImageTransport it(nh);

     #if USE_COMPRESSED_STREAMS
      std::string depth_topic = std::string(fromDepthTopic);
      image_transport::TransportHints hintsDepth("compressedDepth");
 	  image_transport::SubscriberFilter *depth_img_sub = new image_transport::SubscriberFilter();
 	  depth_img_sub->subscribe(it,depth_topic,(uint32_t) 1,hintsDepth);
     #else
	  message_filters::Subscriber<sensor_msgs::Image> *depth_img_sub  = new message_filters::Subscriber<sensor_msgs::Image>(nh,fromDepthTopic,1);
     #endif // USE_COMPRESSED_STREAMS


     #if USE_COMPRESSED_STREAMS
      std::string color_topic = std::string(fromRGBTopic);
      image_transport::TransportHints hints("compressed");
      image_transport::SubscriberFilter *rgb_img_sub = new  image_transport::SubscriberFilter();
      rgb_img_sub->subscribe(it,color_topic, (uint32_t) 1 , hints);
     #else
      message_filters::Subscriber<sensor_msgs::Image> *rgb_img_sub = new  message_filters::Subscriber<sensor_msgs::Image>(nh,fromRGBTopic, 1);
     #endif // USE_COMPRESSED_STREAMS

     std::cerr<<"Done\n";


     #if USE_NONDEFAULT_CALIBRATIONS
       std::cerr<<"Also subscribing to the camera info topics\n";
       message_filters::Subscriber<sensor_msgs::CameraInfo> *depth_cam_info_sub;
       message_filters::Subscriber<sensor_msgs::CameraInfo> *rgb_cam_info_sub;

	   depth_cam_info_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh,fromDepthTopicInfo,1);
	   rgb_cam_info_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh,fromRGBTopicInfo,1);

 	   sync = new message_filters::Synchronizer<RgbdSyncPolicy>(RgbdSyncPolicy(rate), *rgb_img_sub, *depth_img_sub,*depth_cam_info_sub); //*rgb_cam_info_sub,
 	   sync->registerCallback(rgbdCallback);
     #else
       std::cerr<<"Ignoring camera info topics\n";
       sync = new message_filters::Synchronizer<RgbdSyncPolicy>(RgbdSyncPolicy(rate), *rgb_img_sub, *depth_img_sub); //*rgb_cam_info_sub,
	   sync->registerCallback(rgbdCallbackNoCalibration);
    #endif




     std::cerr<<"Waiting for input stream frames\n";
	  while (ros::ok())
		 {
           loopEvent();
		 }


	   delete depth_img_sub;
	   delete rgb_img_sub;


     #if USE_NONDEFAULT_CALIBRATIONS
       delete depth_cam_info_sub;
	   delete rgb_cam_info_sub;
     #endif // USE_NONDEFAULT_CALIBRATIONS

	   delete sync;
	}
	catch(std::exception &e) { ROS_ERROR("Exception: %s", e.what()); return 1; }
	catch(...)               { ROS_ERROR("Unknown Error"); return 1; }
	ROS_ERROR("Shutdown complete");
	return 0;
}


