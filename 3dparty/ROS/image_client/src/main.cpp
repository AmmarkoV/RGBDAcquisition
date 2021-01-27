#include <stdexcept>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>


#include <tf/transform_broadcaster.h>
#include <std_srvs/Empty.h>


#include <iostream>
#include <unistd.h>
#include <ros/ros.h>
#include <ros/spinner.h>


#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


struct calibration
{
  /* CAMERA INTRINSIC PARAMETERS */
  char intrinsicParametersSet;
  float intrinsic[9];
  float k1,k2,p1,p2,k3; 
  
  float cx,cy,fx,fy;
};

unsigned int width=640;
unsigned int height=480;
//Configuration
#define DEFAULT_TF_ROOT "map"
char tfRoot[512]; //This is the string of the root node on the TF tree

unsigned int startTime=0, currentTime=0;

#define camRGBRaw "/camera/rgb/image_rect_color"
#define camRGBInfo "/camera/rgb/camera_info"

sensor_msgs::CameraInfo camInfo;

volatile bool startTrackingSwitch = false;
volatile bool stopTrackingSwitch = false;
volatile int  key=0;

//http://opencv.willowgarage.com/documentation/c/core_utility_and_system_functions_and_macros.html#gettickcount
//The function returns number of the ticks starting from some PLATFORM-DEPENDANT event (number of CPU ticks from the startup, number of milliseconds from 1970th year, etc.)
//Adjust if FPS are crazy :P
#define OpenCVTickMultiplier 1000*1000*1000

//ROS
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo> RgbSyncPolicy;


ros::NodeHandle * nhPtr=0;
//RGB Subscribers
message_filters::Subscriber<sensor_msgs::Image> *rgb_img_sub;
message_filters::Subscriber<sensor_msgs::CameraInfo> *rgb_cam_info_sub; 


//RGBd Callback is called every time we get a new pair of frames , it is synchronized to the main thread
void rgbCallback(const sensor_msgs::Image::ConstPtr rgb_img_msg,const sensor_msgs::CameraInfo::ConstPtr camera_info_msg )
{
    struct calibration intrinsics={0};
  //If it is the first time we are called in this instance , we should get focal length etc
//In order to setup the hand tracker correctly

      ROS_INFO("New frame received..");
      //Using Intrinsic camera matrix for the raw (distorted) input images.

      //Focal lengths presented here are not the same with our calculations ( using zppd zpps )
      //https://github.com/ros-drivers/openni_camera/blob/groovy-devel/src/openni_device.cpp#L197
        intrinsics.fx = camera_info_msg->K[0];
        intrinsics.fy = camera_info_msg->K[4];
        intrinsics.cx = camera_info_msg->K[2];
        intrinsics.cy = camera_info_msg->K[5];

        intrinsics.k1 = camera_info_msg->D[0];
        intrinsics.k2 = camera_info_msg->D[1];
        intrinsics.p1 = camera_info_msg->D[2];
        intrinsics.p2 = camera_info_msg->D[3];
        intrinsics.k3 = camera_info_msg->D[4];

        width = camera_info_msg->width;
        height = camera_info_msg->height;

        printf("We received an initial frame with the following metrics :  width = %u , height = %u , fx %0.2f , fy %0.2f , cx %0.2f , cy %0.2f \n",width,height,intrinsics.fx,intrinsics.fy,intrinsics.cx,intrinsics.cy);

        //cv::Mat rgb = cv::Mat(width,height,cv::CV_8UC3); 
        cv::Mat rgb = cv::Mat::zeros(width,height,CV_8UC3); 
           //A new pair of frames has arrived , copy and convert them so that they are ready
        cv_bridge::CvImageConstPtr orig_rgb_img;
        cv_bridge::CvImageConstPtr orig_depth_img;
        orig_rgb_img = cv_bridge::toCvCopy(rgb_img_msg, "rgb8");
        orig_rgb_img->image.copyTo(rgb);

          //printf("Passing Frame width %u height %u fx %0.2f fy %0.2f cx %0.2f cy %0.2f\n",width,height,intrinsics.fx,intrinsics.fy,intrinsics.cx,intrinsics.cy);
          //ROS_INFO("Passing New Frames to Synergies");
          //passNewFrames((unsigned char*) rgb.data,width , height , &intrinsics , &extrinsics);
          //ROS_INFO("Synergies should now have the new frames");

       camInfo = sensor_msgs::CameraInfo(*camera_info_msg);


    
    cv::Mat rgbTmp = rgb.clone();
    //Take care of drawing stuff as visual output
 cv::Mat bgrMat,rgbMat(height,width,CV_8UC3,rgbTmp.data,3*width);
 cv::cvtColor(rgbMat,bgrMat, cv::COLOR_RGB2BGR);// opencv expects the image in BGR format 
    //After we have our bgr Frame ready and we added the FPS text , lets show it! 
 cv::imshow("MocapNET - RGB input",bgrMat);
 cv::waitKey(1);
    
    return;
    
    
}


void loopEvent()
{/*
  //ROS_INFO("Loop Event started");
  //We spin from this thread to keep thread synchronization problems at bay
  frameSpinner();

  // if we got a depth and rgb frames , lets go

  // The score is a negative number indicating the quality of the result. Higher is better. The best score is 0.
        //Update our TF root string from the ROS parameter ( possibly changed )
  std::string tmp;
        if (nhPtr->getParamCached("trackerPoseParentTFNode", tmp)) { strcpy(tfRoot,(char*) tmp.c_str()); }

        key=getKeyPressed();
        //Post our geometry to ROS
        geometry_msgs::Pose str;
  updatePose(str);

  if ( weHaveRGBAndDepthFrames() )
   {
        //If we draw out we have to visualize the hand pose , set up windows , put out text etc.
        if (draw_out)
                {
                    doDrawOut();
                } //End of drawing code..

     }

     //Alternate between On And Off states at tracking
 if (
         ((stopTrackingSwitch) && (trackingState)  )  ||
         ((startTrackingSwitch) && (!trackingState))  ||
         (key=='s')
        )
    {
     if ( !trackingState ) { trackingState=true; } else  { trackingState=false; }
     // reset the starting pose of the tracker to initialization position.
     startTrackingSwitch=false;
     stopTrackingSwitch=false;
    }
 // <- end of we have depth and rgb code
 */
}



int main(int argc, char **argv)
{
   ROS_INFO("Initializing MocapNET ROS Wrapper");
   try
 {
    ros::init(argc, argv, "MocapNET");
     ros::start();

    //ros::AsyncSpinner spinner(0); <- We dont use an asynchronous spinner any more
     ros::NodeHandle nh;
     nhPtr = &nh;

     message_filters::Synchronizer<RgbSyncPolicy> *sync;
 
  rgb_img_sub = new  message_filters::Subscriber<sensor_msgs::Image>(nh,camRGBRaw, 1);
  rgb_cam_info_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh,camRGBInfo,1);


   sync = new message_filters::Synchronizer<RgbSyncPolicy>(RgbSyncPolicy(5), *rgb_img_sub,*rgb_cam_info_sub);  
     //rosrun rqt_graph rqt_graph to test out whta is going on


     //We advertise the services we want accessible using "rosservice call *w/e*"
/*
      ros::ServiceServer startTrackingService = nh.advertiseService("startTracking", startTracking);
      ros::ServiceServer stopTrackingService = nh.advertiseService("stopTracking", stopTracking);
      ros::ServiceServer startDrawingService = nh.advertiseService("startDrawing", startDrawing);
      ros::ServiceServer stopDrawingService = nh.advertiseService("stopDrawing", stopDrawing);
      ros::ServiceServer setExtrinsicsService = nh.advertiseService("setExtrinsics", setExtrinsics);

      //These are the parameters we use , and their default values "rosparam get/set *w/e*"
      nh.setParam("trackerPoseScore",0);
      nh.setParam("trackerPoseParentTFNode", DEFAULT_TF_ROOT);
      strcpy(tfRoot,DEFAULT_TF_ROOT);
*/

   sync->registerCallback(rgbCallback); 
      //registerResultCallback((void*) sampleResultingSynergies);
      //registerUpdateLoopCallback((void *) loopEvent);
      //registerROSSpinner( (void *) frameSpinner );

      ROS_INFO("We should be able to process incoming messages now!");


      //---------------------------------------------------------------------------------------------------
      //This code segment waits for a valid first frame to come and initialize the focal lengths etc..
      //If there is no first frame it times out after a little while displaying a relevant error message
      //---------------------------------------------------------------------------------------------------
      //if (!weCanReceiveImages()) { ROS_ERROR("Stopping service.."); return 1; }

      //Create our context
      //---------------------------------------------------------------------------------------------------

   //////////////////////////////////////////////////////////////////////////

      unsigned long i = 0;
     // startTime=cvGetTickCount();
   while ( ( key!='q' ) && (ros::ok()) )
  {
          ros::spinOnce();
          if (i%1000==0)
          {
            fprintf(stderr,".");
            loopEvent(); //<- this keeps our ros node messages handled up until synergies take control of the main thread
          }
          ++i;
 

          usleep(1000);
   }

 }
 catch(std::exception &e) { ROS_ERROR("Exception: %s", e.what()); return 1; }
 catch(...)               { ROS_ERROR("Unknown Error"); return 1; }


   //switchDrawOutTo(0);
   ROS_ERROR("Shutdown complete");
   return 0;
}
