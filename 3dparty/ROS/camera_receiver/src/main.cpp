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

#include "AmmClient.h"


 
#include <unistd.h>


#define NODE_NAME "camera_receiver"



#define EMMIT_CALIBRATION 0
#define USE_CALIBRATION 0






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
char tfRoot[512]= {"map"};

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

volatile static bool first = true;

int width = 0;
int height = 0;

unsigned int draw_out = 0;
unsigned int counter = 0;
char filecontent[1024 * 1024 * 1]= {0};
unsigned int filecontentMaxSize=1024 * 1024 * 1;


struct AmmClient_Instance * connection=0;

#if USE_CALIBRATION
struct calibration calibRGB;
struct calibration calibDepth;
#endif

enum FRAMERATE_STATES
{
    STOPPED_FRAMERATE = 0,
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


int doDrawOutFrame( unsigned char * rgbFrame, unsigned int rgbWidth, unsigned int rgbHeight,
                    unsigned short * depthFrame, unsigned int depthWidth, unsigned int depthHeight )
{
    if (!draw_out)
    {
        return 0;
    }
    //-----------------------------------------------------------------------------------------------------------------------
    cv::Mat wrappedColor(rgbHeight,rgbWidth, CV_8UC3,  rgbFrame, rgbWidth*3);  // does not copy
    cv::Mat rgbTmp = wrappedColor.clone();
    //Take care of drawing stuff as visual output
    cv::Mat bgrMat,rgbMat(rgbHeight,rgbWidth,CV_8UC3,rgbTmp.data,3*rgbWidth);
    cv::cvtColor(rgbMat,bgrMat, CV_RGB2BGR);// opencv expects the image in BGR format


    //-----------------------------------------------------------------------------------------------------------------------
    cv::Mat wrappedDepth(depthHeight,depthWidth, CV_16UC1,depthFrame, depthWidth*2);  // does not copy

    cv::Mat depthNorm;
    cv::normalize(wrappedDepth,depthNorm,0,255,CV_MINMAX,CV_8UC1);

    //After we have our bgr Frame ready and we added the FPS text , lets show it!
    cv::imshow("Depth",depthNorm);
    cv::imshow("RGB",bgrMat);

    return 1;
}


int doDrawOut(  )
{
    // return doDrawOutFrame(acquisitionGetColorFrame(moduleID,devID),colorWidth,colorHeight,acquisitionGetDepthFrame(moduleID,devID),depthWidth,depthHeight);
    return 0;
}


int getKeyPressed()
{
    return cv::waitKey(3)&0x000000FF;
    return ' '; /*No Windows -> No keys :P */
}



bool publishImagesFrames(cv::Mat & matImg)
{
    //We want to populate this
    ros::Time sampleTime = ros::Time::now();
    cv_bridge::CvImage out_RGB_msg;
    out_RGB_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
    out_RGB_msg.header.frame_id= tfRoot;
    out_RGB_msg.header.stamp= sampleTime;


    //convert & publish RGB Stream - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //IplImage *imageRGB = cvCreateImageHeader( cvSize(colorWidth,colorHeight), IPL_DEPTH_8U ,3);
    //imageRGB->imageData = (char *) color;
    //out_RGB_msg.header   = in_msg->header; // Same timestamp and tf frame as input image

    //cv::Mat imageRGBMat(cv::Size(colorWidth, colorHeight), CV_8UC3, (char *) color, cv::Mat::AUTO_STEP);
    out_RGB_msg.image = matImg; // Your cv::Mat
    pubRGB.publish(out_RGB_msg.toImageMsg());




    //---------------------------------------------------------------------------------------------------
    //  CAMERA INFORMATION BROADCAST --------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------
#if EMMIT_CALIBRATION
    if (calib!=0)
    {
        int i=0;

        sensor_msgs::CameraInfo cal;
        cal.header.stamp = sampleTime;
        cal.width=colorWidth;
        cal.height=colorHeight;

        cal.D.resize(5, 0.0);
        cal.distortion_model=sensor_msgs::distortion_models::PLUMB_BOB;
        cal.D[0]=calib->k1;
        cal.D[1]=calib->k2;
        cal.D[2]=calib->p1;
        cal.D[3]=calib->p2;
        cal.D[4]=calib->k3;

        for (i=0; i<9; i++)
        {
            cal.K[i]=calib->intrinsic[i];
        }

        cal.R[0]=1.0;
        cal.R[1]=0.0;
        cal.R[2]=0.0;
        cal.R[3]=0.0;
        cal.R[4]=1.0;
        cal.R[5]=0.0;
        cal.R[6]=0.0;
        cal.R[7]=0.0;
        cal.R[8]=1.0;

        for (i=0; i<12; i++)
        {
            cal.P[i]=0.0;
        }

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



//----------------------------------------------------------


void loopEvent(std::string URI)
{
    //ROS_INFO("Loop Event started");
    //We spin from this thread to keep thread synchronization problems at bay
    ros::spinOnce();

    unsigned int filecontentSize = filecontentMaxSize;
    if (
        AmmClient_RecvFile(
            connection,
            URI.c_str(),
            filecontent,
            &filecontentSize,
            1,// int keepAlive,
            0//int reallyFastImplementation
        )
    )
    {
        usleep(100000);
        if (framerateState!=STOPPED_FRAMERATE)
        {
            char * jpegFile = AmmClient_seekEndOfHeader(filecontent,&filecontentSize);
            if (jpegFile!=0)
            {
                cv::Mat matImg  = cv::imdecode(cv::Mat(1, filecontentSize, CV_8UC1, jpegFile),cv::IMREAD_UNCHANGED);

                publishImagesFrames(matImg);

                cv::Mat bgrMat;
                cv::cvtColor(matImg,bgrMat, CV_RGB2BGR);// opencv expects the image in BGR format

                cv::imshow("RGB",bgrMat);
                key=getKeyPressed(); //Post our geometry to ROS


            }
            else
            {
                std::cerr<<"Received something but not an image\n";
            }
        }

        // if we got a depth and rgb frames , lets go
        //If we draw out we have to visualize the hand pose , set up windows , put out text etc.
    }
    else
    {
        std::cerr<<"Failed receiving new image \n";
    }
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
        nhPtr = &nh;

        std::string name;
        std::string camera;
        std::string URI;
        std::string frame;
        std::string server;
        int rate,port,timeout;

        ros::NodeHandle private_node_handle_("~");
        private_node_handle_.param("name", name, std::string("camera_receiver"));
        private_node_handle_.param("camera", camera, std::string("camera/color/"));
        private_node_handle_.param("frame", frame, std::string("frame"));
        private_node_handle_.param("server", server, std::string("139.91.185.16"));
        private_node_handle_.param("URI",URI, std::string("/stream/uploads/image.jpg"));
        private_node_handle_.param("port", port, int(80));
        private_node_handle_.param("timeout", timeout, int(10));


        private_node_handle_.param("width", width, int(640));
        private_node_handle_.param("height", height, int(480));
        private_node_handle_.param("rate", rate, int(10));
        //private_node_handle_.param("scaleDepth", scaleDepth, float(1.0));

        //Pass root frame for TF poses
        strcpy(tfRoot,frame.c_str());

        fprintf(stderr,"Initialized..\n");

        std::cout<<"camera_receiver starting settings ----------------"<<std::endl;

        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL)
            std::cout<<"Current working dir : " << cwd << std::endl;

        std::cout<<"Server : "<<server<<std::endl;
        std::cout<<"Port : "<<port<<std::endl;
        std::cout<<"Name : "<<name<<std::endl;
        std::cout<<"Camera : "<<camera<<std::endl;
        std::cout<<"Frame : "<<frame<<std::endl;
        std::cout<<"TF Root : "<<tfRoot<<std::endl;
        std::cout<<"Mode : "<<width<<"x"<<height<<":"<<rate<<std::endl;
        std::cout<<"--------------------------------------------------"<<std::endl;


        ros::Rate loopRate(rate); //  hz should be our target performance
        framerateState=HIGH_FRAMERATE; //By default we go to medium!

        //We advertise the services we want accessible using "rosservice call *w/e*"
        //ros::ServiceServer saveROSCalibrationService  = nh.advertiseService(name+"/saveROSCalibration", saveROSCalibrationFile);
        ros::ServiceServer visualizeOnService         = nh.advertiseService(name+"/visualize_on", visualizeOn);
        ros::ServiceServer visualizeOffService        = nh.advertiseService(name+"/visualize_off", visualizeOff);
        ros::ServiceServer terminateService           = nh.advertiseService(name+"/terminate", terminate);
        ros::ServiceServer pauseService               = nh.advertiseService(name+"/pause", pause);
        ros::ServiceServer resumeService              = nh.advertiseService(name+"/resume", resume);
        //ros::ServiceServer setScaleService            = nh.advertiseService(name+"/setScale", setScale);

        //Framerate switches
        ros::ServiceServer setStoppedFramerateService   = nh.advertiseService(name+"/setStoppedFramerate", setStoppedFramerate);
        ros::ServiceServer setNormalFramerateService    = nh.advertiseService(name+"/setNormalFramerate", setMediumFramerate);


        //Output RGB Image
        image_transport::ImageTransport it(nh);

        pubRGB = it.advertise(camera+"image_rect_color", 1);
        pubRGBInfo = nh.advertise<sensor_msgs::CameraInfo>(camera+"camera_info",1);
        std::cout<<"ROS RGB Topic name : "<<camera<<"image_rect_color"<<std::endl;


        //---------------------------------------------------------------------------------------------------
        //This code segment waits for a valid first frame to come and initialize the focal lengths etc..
        //If there is no first frame it times out after a little while displaying a relevant error message
        //---------------------------------------------------------------------------------------------------


        std::cout<<"Trying to open capture device.."<<std::endl;
        ROS_INFO("Trying to open capture device..");
        connection = AmmClient_Initialize(server.c_str(),port,timeout/*sec*/); 
        if(connection==0)
        {
            std::cerr<<"Could not establish connection to "<<server<<":"<<port<<std::endl;
            return 0;
        }

        ROS_INFO("Done Initializing camera_receiver , now entering grab loop..");
        while ( ( key!='q' ) && (ros::ok()) )
        {
            loopEvent(URI); //<- this keeps our ros node messages handled up until synergies take control of the main thread
            loopRate.sleep();
        }

        switchDrawOutTo(0);
    }
    catch(std::exception &e)
    {
        ROS_ERROR("Exception: %s", e.what());
        return 1;
    }
    catch(...)
    {
        ROS_ERROR("Unknown Error");
        return 1;
    }
    ROS_ERROR("Shutdown complete");
    return 0;
}
