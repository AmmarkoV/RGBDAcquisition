#include <stdexcept>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

#include <std_msgs/Float32MultiArray.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
//#include <tf/transform_broadcaster.h>
#include <std_srvs/Empty.h>


#include <math.h>
#include <iostream>
#include <unistd.h>
#include <ros/ros.h>
#include <ros/spinner.h>


#include "AmmClient.h"
#include "jpgInput.h"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


struct AmmClient_Instance * connection = 0;

#define USE_CURL 0

unsigned long maxCompressedJPEGFile = 1024* 1024 * 1; //2MB
unsigned long currentCompressedJPEGFile = 0;
char * compressedJPEGFile = 0;

unsigned int frameID=0;

struct calibration
{
    /* CAMERA INTRINSIC PARAMETERS */
    char intrinsicParametersSet;
    float intrinsic[9];
    float k1,k2,p1,p2,k3;

    float cx,cy,fx,fy;
};



#define camRGBRaw "/camera/rgb/image_rect_color"
#define camRGBInfo "/camera/rgb/camera_info"
#define DEFAULT_TF_ROOT "map"
unsigned int width=640;
unsigned int height=480;

int useSimpleBroadcaster=0;

//Configuration
char tfRoot[512]= {DEFAULT_TF_ROOT}; //This is the string of the root node on the TF tree

int publishCameraTF=1;
float cameraXPosition=0.0,cameraYPosition=0.0,cameraZPosition=0.0;
float cameraRoll=90.0, cameraPitch=0.0, cameraYaw=0.0;
unsigned int startTime=0, currentTime=0;


sensor_msgs::CameraInfo camInfo;

volatile bool startTrackingSwitch = false;
volatile bool stopTrackingSwitch = false;
volatile int  key=0;

//ROS
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo> RgbSyncPolicy;

ros::NodeHandle * nhPtr=0;
//RGB Subscribers
message_filters::Subscriber<sensor_msgs::Image> *rgb_img_sub;
message_filters::Subscriber<sensor_msgs::CameraInfo> *rgb_cam_info_sub;

bool visualizeAngles(std_srvs::Empty::Request& request,std_srvs::Empty::Response& response)
{
    ROS_INFO("Visualize Angles called");
    return true;
}

bool visualizeMain(std_srvs::Empty::Request& request,std_srvs::Empty::Response& response)
{
    ROS_INFO("Visualize Main called");
    return true;
}

bool visualizeOverlay(std_srvs::Empty::Request& request,std_srvs::Empty::Response& response)
{
    ROS_INFO("Visualize Overlay called");
    return true;
}

bool visualizeOff(std_srvs::Empty::Request& request,std_srvs::Empty::Response& response)
{
    ROS_INFO("Visualize Off called");
    return true;
}

bool terminate(std_srvs::Empty::Request& request,std_srvs::Empty::Response& response)
{
    ROS_INFO("Terminating MocapNET");
    exit(0);
    return true;
}



float * mallocVectorR(std::vector<float> bvhFrame)
{
    if (bvhFrame.size()==0)
    {
        fprintf(stderr,"mallocVector given an empty vector..\n");
        //Empty bvh frame means no vector
        return 0;
    }

    float * newVector = (float*) malloc(sizeof(float) * bvhFrame.size());
    if (newVector!=0)
    {
        for (int i=0; i<bvhFrame.size(); i++)
        {
            newVector[i]=(float) bvhFrame[i];
        }
    }
    return newVector;
}



//RGB Callback is called every time we get a new frame, it is synchronized to the main thread
void rgbCallback(const sensor_msgs::Image::ConstPtr rgb_img_msg,const sensor_msgs::CameraInfo::ConstPtr camera_info_msg)
{
    ++frameID;
    struct calibration intrinsics= {0};

    //ROS_INFO("New frame received..");
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

    //printf("We received an initial frame with the following metrics :  width = %u , height = %u , fx %0.2f , fy %0.2f , cx %0.2f , cy %0.2f \n",width,height,intrinsics.fx,intrinsics.fy,intrinsics.cx,intrinsics.cy);

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
    cv::imshow("RGB input",bgrMat);

    struct Image pic= {0};
    pic.pixels = rgb.data ;
    pic.width  = rgb.size().width;
    pic.height = rgb.size().height;
    pic.channels = 3;
    pic.bitsperpixel = 24;
    pic.image_size =pic.width * pic.height *pic.channels ;

#if USE_CURL
    if (frameID%10==0)
    {
        fprintf(stderr,"flushing to disk\n");
        WriteJPEGFile(&pic,"/home/dji/catkin_ws/src/camera_broadcast/src/image.jpg");
        int i=system("curl -F \"submit=1\" -F \"fileToUpload=@/home/dji/catkin_ws/src/camera_broadcast/src/image.jpg\" ammar.gr/stream/upload.php");
        usleep(10000);
    }
#else
    if (connection)
    {
        currentCompressedJPEGFile = maxCompressedJPEGFile;
        if ( WriteJPEGMemory(&pic,compressedJPEGFile,&currentCompressedJPEGFile) )
        {
            if (
                AmmClient_SendFile(
                    connection,
                    "/stream/upload.php",
                    "fileToUpload",
                    "image.jpg",
                    "image/jpeg",
                    compressedJPEGFile,
                    (unsigned int) currentCompressedJPEGFile,
                    1
                )
            )
            {
                fprintf(stderr,"* %lu",currentCompressedJPEGFile);
                //usleep(1000000);
                char buf[4098]= {0};
                unsigned int recvdSize=4098;

                if (!AmmClient_Recv(connection,buf,&recvdSize) )
                {
                    fprintf(stderr,RED "Failed to recv.. \n" NORMAL);
                }

                fprintf(stderr,"Response = `%s`\n",buf);
            } else
            {
                fprintf(stderr,RED "Unable to do HTTP transmission \n" NORMAL);
            }

        } else
        {
            fprintf(stderr, RED "Unable to do JPEG compression \n" NORMAL);
        }
    }
#endif


    cv::waitKey(1);

    return;
}




int main(int argc, char **argv)
{
    //https://github.com/AmmarkoV/RGBDAcquisition/tree/master/3dparty/ROS/rgbd_acquisition
    //==================================================================================================================================
    //roslaunch rgbd_acquisition rgb_acquisition.launch deviceID:=sven.mp4-data moduleID:=TEMPLATE width:=1920 height:=1080 framerate:=1
    //roslaunch camera_broadcast camera_broadcast.launch
    //==================================================================================================================================


    compressedJPEGFile =  (char*) malloc(sizeof(char)* maxCompressedJPEGFile);
    if (compressedJPEGFile==0)
    {
        ROS_INFO("Could not allocate memory..!");
        exit(1);
    }

    connection = AmmClient_Initialize("139.91.185.16",80,10/*sec*/);


    ROS_INFO("Initializing MocapNET ROS Wrapper");
    try
    {
        ros::init(argc, argv, "MocapNET");
        ros::start();

        ros::NodeHandle nh;
        ros::NodeHandle private_node_handle("~");
        nhPtr = &nh;

        float rate = 5;
        std::string joint2DEstimatorName;
        std::string name;
        std::string fromRGBTopic;
        std::string fromRGBTopicInfo;
        std::string tfRootName;
        std::string tfTargetBVHFilename;

        ROS_INFO("Initializing Parameters..");

        private_node_handle.param("tfTargetBVHFilename", tfTargetBVHFilename, std::string("dataset/headerWithHeadAndOneMotion.bvh"));
        private_node_handle.param("fromRGBTopic", fromRGBTopic, std::string(camRGBRaw));
        private_node_handle.param("fromRGBTopicInfo", fromRGBTopicInfo, std::string(camRGBInfo));
        private_node_handle.param("name", name, std::string("camera_broadcast"));
        private_node_handle.param("rate",rate);


        private_node_handle.param("tfRoot",tfRootName, std::string(DEFAULT_TF_ROOT));
        snprintf(tfRoot,510,"%s",tfRootName.c_str());
        fprintf(stderr,"TFRoot Name = %s ",tfRoot);


        rgb_img_sub = new  message_filters::Subscriber<sensor_msgs::Image>(nh,fromRGBTopic, 1);
        rgb_cam_info_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh,fromRGBTopicInfo,1);
        message_filters::Synchronizer<RgbSyncPolicy> *sync = new message_filters::Synchronizer<RgbSyncPolicy>(RgbSyncPolicy(1), *rgb_img_sub,*rgb_cam_info_sub);
        //rosrun rqt_graph rqt_graph to test out what is going on

        ros::ServiceServer visualizeAnglesService    = nh.advertiseService(name + "/visualize_angles",&visualizeAngles);
        ros::ServiceServer visualizeMainService      = nh.advertiseService(name + "/visualize_main",&visualizeMain);
        ros::ServiceServer visualizeOverlayService   = nh.advertiseService(name + "/visualize_overlay",&visualizeOverlay);
        ros::ServiceServer visualizeOffService       = nh.advertiseService(name + "/visualize_off", &visualizeOff);
        ros::ServiceServer terminateService          = nh.advertiseService(name + "/terminate", terminate);

        //registerResultCallback((void*) sampleResultingSynergies);
        //registerUpdateLoopCallback((void *) loopEvent);
        //registerROSSpinner( (void *) frameSpinner );


        ROS_INFO("Done with ROS initialization!");


        ros::Rate rosrate(rate);

        ROS_INFO("Initializing 2D joint estimator");
        if ( 1 )
        {
            ROS_INFO("Initializing Camera Broadcast");
            if ( 1 )
            {
                //Last check for inconsistencies..
                //Since the tfTargetBVHFilename can load a different BVH file than the internal model
                //We need to check that the two BVH armatures have joint parity..
                //A stricter test could also make sure joint names are the same, however someone can essencially alter the Frame names using this file
                //so we will only check for the number of joints..!


                sync->registerCallback(rgbCallback);
                ROS_INFO("Registered ROS services, we should be able to process incoming messages now!");

                //Lets go..
                //////////////////////////////////////////////////////////////////////////
                unsigned long i = 0;
                // startTime=cvGetTickCount();
                while ( ( key!='q' ) && (ros::ok()) )
                {

                    if (i%1000==0)
                    {
                        fprintf(stderr,".");
                    }
                    ++i;

                    //usleep(1000);
                    ros::spinOnce();
                    rosrate.sleep();
                }
                //////////////////////////////////////////////////////////////////////////
            } else
            {
                ROS_ERROR("Failed to initialize MocapNET 3D pose estimation..");
            }
        } else
        {
            ROS_ERROR("Failed to initialize MocapNET built-in 2D joint estimation..");
        }

    }
    catch(std::exception &e) {
        ROS_ERROR("Exception: %s", e.what());
        return 1;
    }
    catch(...)               {
        ROS_ERROR("Unknown Error");
        return 1;
    }

    AmmClient_Close(connection);

    ROS_INFO("Shutdown complete");
    return 0;
}
