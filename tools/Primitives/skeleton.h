#ifndef SKELETON_H_INCLUDED
#define SKELETON_H_INCLUDED

#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif



static float defaultJoints2D[] = { 293.0, 37.0, 293.0, 100.0, 292.0, 170.0, 235.0, 99.0, 349.0, 101.0, 172.0, 144.0, 421.0, 128.0, 113.0, 176.0, 480.0, 144.0, 263.0, 242.0, 319.0, 242.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 291.0, 242.0 };
static float defaultJoints[] = { -88.01861572265625, -655.5640258789062, 1717.0, -91.15103149414062, -464.8657531738281, 1764.83154296875, -93.02406311035156, -229.4916534423828, 1740.753173828125, -281.5615234375, -465.376708984375, 1757.0, 98.7568359375, -467.6025390625, 1785.0, -498.959716796875, -323.6999206542969, 1784.90966796875, 364.5001220703125, -404.3070373535156, 1923.611572265625, -673.8611450195312, -208.03506469726562, 1731.8184814453125, 562.7467651367188, -339.4126281738281, 1868.3612060546875, -185.67507934570312, 5.257970809936523, 1733.0770263671875, -4.1191253662109375, 6.506894111633301, 1700.2725830078125, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -94.89710235595703, 5.882432460784912, 1716.6748046875 } ;



#define dp 1700
#define sx 320
#define sy 100                        //head         neck         bodycenter   rightshoulder  leftshoulder     rightElbow      leftElbow       rightWrist    leftWrist
static float NAOdefaultJoints2D[] = { sx+0,sy-94 , sx+0,sy+0    , sx+0,sy+22  , sx-191,sy+0 ,  sx+191,sy+0  , sx-191,sy+0   , sx+191,sy+0   , sx-191,sy+0 , sx+191,sy+0 ,
                                      //rightLegRoot  leftLegRoot    rightKnee      leftKnee     rightAnckle      LeftAnckle    hip
                                      sx-80,sy+122 , sx+80,sy+122,  sx-80,sy+200, sx+80,sy+200,  sx-80,sy+300 , sx+80,sy+300 , sx+0,sy+122
                                    };
static float NAOdefaultJoints[] = { 0,-194,dp, 0,0,dp , 0,222,dp , -191,0,dp , 191,0,dp , -191,0,dp-200 , 191,0,dp-200 ,
                                    -191,0,dp-700 , 101,0,dp-700 ,  -80,322,dp , 80,322,dp  , -80,700,dp , 80,700,dp , -80,1000,dp , 80,1000,dp ,
                                    0,322,dp
                                  };



static const char * smartBodyNames[] =
{
    "JtUpperFaceParent",
    "JtNeckB" ,
    "JtSpineB",
    "JtShoulderRt",
    "JtShoulderLf",
    "JtUpperArmTwistBRt",
    "JtUpperArmTwistBLf",
    "JtWristRt",
    "JtWristLf",
    "JtHipRt",
    "JtHipLf",
    "JtKneeRt",
    "JtKneeLf",
    "JtAnkleRt",
    "JtAnkleLf" ,
    "JtSpineA" ,
//=================
    "End of Joint Names"
};


static const char * jointNames[] =
{
    "head",
    "neck" ,
    "torso",
    "right_shoulder",
    "left_shoulder",
    "right_elbow",
    "left_elbow",
    "right_hand",
    "left_hand",
    "right_hip",
    "left_hip",
    "right_knee",
    "left_knee",
    "right_foot",
    "left_foot" ,
    "hip" ,
//=================
    "End of Joint Names"
};


static const char * tgbtNames[] =
{
    "head",
    "neck",
    "bodyCenter",
    "rightShoulder",
    "leftShoulder",
    "rightElbow",
    "leftElbow",
    "rightWrist",
    "leftWrist",
    "rightLegRoot",
    "leftLegRoot",
    "rightKnee",
    "leftKnee",
    "rightAnkle",
    "leftAnkle",
    "hip",
//--------------------
    "End Of TGBT Names"
};

static const char * const humanSkeletonJointNames[] =
{
    "HUMAN_SKELETON_HEAD",
    "HUMAN_SKELETON_NECK",
    "HUMAN_SKELETON_TORSO",
    "HUMAN_SKELETON_RIGHT_SHOULDER",
    "HUMAN_SKELETON_LEFT_SHOULDER",
    "HUMAN_SKELETON_RIGHT_ELBOW",
    "HUMAN_SKELETON_LEFT_ELBOW",
    "HUMAN_SKELETON_RIGHT_HAND",
    "HUMAN_SKELETON_LEFT_HAND",
    "HUMAN_SKELETON_RIGHT_HIP",
    "HUMAN_SKELETON_LEFT_HIP",
    "HUMAN_SKELETON_RIGHT_KNEE",
    "HUMAN_SKELETON_LEFT_KNEE",
    "HUMAN_SKELETON_RIGHT_FOOT",
    "HUMAN_SKELETON_LEFT_FOOT",
    "HUMAN_SKELETON_HIP"
};

enum humanSkeletonJoints
{
    HUMAN_SKELETON_HEAD = 0,
    HUMAN_SKELETON_NECK,
    HUMAN_SKELETON_TORSO,
    HUMAN_SKELETON_RIGHT_SHOULDER,
    HUMAN_SKELETON_LEFT_SHOULDER,
    HUMAN_SKELETON_RIGHT_ELBOW,
    HUMAN_SKELETON_LEFT_ELBOW,
    HUMAN_SKELETON_RIGHT_HAND,
    HUMAN_SKELETON_LEFT_HAND,
    HUMAN_SKELETON_RIGHT_HIP,
    HUMAN_SKELETON_LEFT_HIP,
    HUMAN_SKELETON_RIGHT_KNEE,
    HUMAN_SKELETON_LEFT_KNEE,
    HUMAN_SKELETON_RIGHT_FOOT,
    HUMAN_SKELETON_LEFT_FOOT,
    HUMAN_SKELETON_HIP,
    //---------------------
    HUMAN_SKELETON_PARTS
};


static const int humanSkeletonJointsRelationMap[] =
{
    // Parent                        Joint
    HUMAN_SKELETON_HEAD,           //HUMAN_SKELETON_HEAD
    HUMAN_SKELETON_HEAD,           //HUMAN_SKELETON_NECK
    HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_TORSO
    HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_RIGHT_SHOULDER
    HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_LEFT_SHOULDER
    HUMAN_SKELETON_RIGHT_SHOULDER, //HUMAN_SKELETON_RIGHT_ELBOW
    HUMAN_SKELETON_LEFT_SHOULDER,  //HUMAN_SKELETON_LEFT_ELBOW
    HUMAN_SKELETON_RIGHT_ELBOW,    //HUMAN_SKELETON_RIGHT_HAND
    HUMAN_SKELETON_LEFT_ELBOW,     //HUMAN_SKELETON_LEFT_HAND
    HUMAN_SKELETON_HIP,            //HUMAN_SKELETON_RIGHT_HIP
    HUMAN_SKELETON_HIP,            //HUMAN_SKELETON_LEFT_HIP
    HUMAN_SKELETON_RIGHT_HIP,      //HUMAN_SKELETON_RIGHT_KNEE
    HUMAN_SKELETON_LEFT_HIP,       //HUMAN_SKELETON_LEFT_KNEE
    HUMAN_SKELETON_LEFT_KNEE,      //HUMAN_SKELETON_RIGHT_FOOT
    HUMAN_SKELETON_LEFT_KNEE,      //HUMAN_SKELETON_LEFT_FOOT
    HUMAN_SKELETON_TORSO,          //HUMAN_SKELETON_HIP
};


static const char * const humanSkeletonMirroredJointNames[] =
{
    "HUMAN_SKELETON_MIRRORED_HEAD",
    "HUMAN_SKELETON_MIRRORED_NECK",
    "HUMAN_SKELETON_MIRRORED_TORSO",
    "HUMAN_SKELETON_MIRRORED_LEFT_SHOULDER",
    "HUMAN_SKELETON_MIRRORED_RIGHT_SHOULDER",
    "HUMAN_SKELETON_MIRRORED_LEFT_ELBOW",
    "HUMAN_SKELETON_MIRRORED_RIGHT_ELBOW",
    "HUMAN_SKELETON_MIRRORED_LEFT_HAND",
    "HUMAN_SKELETON_MIRRORED_RIGHT_HAND",
    "HUMAN_SKELETON_MIRRORED_LEFT_HIP",
    "HUMAN_SKELETON_MIRRORED_RIGHT_HIP",
    "HUMAN_SKELETON_MIRRORED_LEFT_KNEE",
    "HUMAN_SKELETON_MIRRORED_RIGHT_KNEE",
    "HUMAN_SKELETON_MIRRORED_LEFT_FOOT",
    "HUMAN_SKELETON_MIRRORED_RIGHT_FOOT",
    "HUMAN_SKELETON_MIRRORED_HIP"
};

enum humanMirroredSkeletonJoints
{
    HUMAN_SKELETON_MIRRORED_HEAD = 0,
    HUMAN_SKELETON_MIRRORED_NECK,
    HUMAN_SKELETON_MIRRORED_TORSO,
    HUMAN_SKELETON_MIRRORED_LEFT_SHOULDER,
    HUMAN_SKELETON_MIRRORED_RIGHT_SHOULDER,
    HUMAN_SKELETON_MIRRORED_LEFT_ELBOW,
    HUMAN_SKELETON_MIRRORED_RIGHT_ELBOW,
    HUMAN_SKELETON_MIRRORED_LEFT_HAND,
    HUMAN_SKELETON_MIRRORED_RIGHT_HAND,
    HUMAN_SKELETON_MIRRORED_LEFT_HIP,
    HUMAN_SKELETON_MIRRORED_RIGHT_HIP,
    HUMAN_SKELETON_MIRRORED_LEFT_KNEE,
    HUMAN_SKELETON_MIRRORED_RIGHT_KNEE,
    HUMAN_SKELETON_MIRRORED_LEFT_FOOT,
    HUMAN_SKELETON_MIRRORED_RIGHT_FOOT,
    HUMAN_SKELETON_MIRRORED_HIP,
    //---------------------
    HUMAN_SKELETON_MIRRORED_PARTS
};


struct point2D
{
    float x,y;
};

struct point3D
{
    float x,y,z;
};




struct skeletonNAO
{
    unsigned int observationNumber , observationTotal;
    unsigned int userID;

    unsigned char isNew,isVisible,isOutOfScene,isLost;
    unsigned char statusCalibrating,statusStoppedTracking, statusTracking,statusFailed;

    struct point3D bbox[8];
    struct point3D bboxDimensions;
    struct point3D centerOfMass;
    struct point3D joint[HUMAN_SKELETON_PARTS];
    struct point2D joint2D[HUMAN_SKELETON_PARTS];
    float jointAccuracy[HUMAN_SKELETON_PARTS];
};




struct skeletonHuman
{
    unsigned int observationNumber , observationTotal;
    unsigned int userID;

    unsigned char isNew,isVisible,isOutOfScene,isLost;
    unsigned char statusCalibrating,statusStoppedTracking, statusTracking,statusFailed;

    struct point3D bbox[8];
    struct point3D bboxDimensions;
    struct point3D centerOfMass;
    struct point3D joint[HUMAN_SKELETON_PARTS];
    struct point2D joint2D[HUMAN_SKELETON_PARTS];
    float  jointAccuracy[HUMAN_SKELETON_PARTS];
    unsigned int active[HUMAN_SKELETON_PARTS];

    struct point3D relativeJointAngle[HUMAN_SKELETON_PARTS];
};


struct skeletonPointing
{
    struct point3D pointStart;
    struct point3D pointEnd;
    struct point3D pointingVector;
    unsigned char isLeftHand;
    unsigned char isRightHand;
};


static int cleanSkeleton(struct skeletonHuman * sk)
{
    unsigned int i=0;

    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        sk->relativeJointAngle[i].x=0.0;
        sk->relativeJointAngle[i].y=0.0;
        sk->relativeJointAngle[i].z=0.0;
        sk->joint[i].x=0.0;
        sk->joint[i].y=0.0;
        sk->joint[i].z=0.0;
        sk->joint2D[i].x=0.0;
        sk->joint2D[i].y=0.0;
        sk->jointAccuracy[i]=0.0;
    }

    return 1;
}


static int skeletonEmpty3DJoint(struct skeletonHuman * sk , unsigned int j1)
{
    //Check NotSet
    if (
        (sk->joint[j1].x==0) &&
        (sk->joint[j1].y==0) &&
        (sk->joint[j1].z==0)
    )
    {
        return 1;
    }


    //Check NaN
    if (
        (sk->joint[j1].x!=sk->joint[j1].x) ||
        (sk->joint[j1].y!=sk->joint[j1].y) ||
        (sk->joint[j1].z!=sk->joint[j1].z)
    )
    {
        return 1;
    }

    return 0;
}


static int skeletonEmpty2DJoint(struct skeletonHuman * sk , unsigned int j1)
{
    //Check NotSet
    if (
        (sk->joint2D[j1].x==0) &&
        (sk->joint2D[j1].y==0)
    )
    {
        return 1;
    }


    //Check NaN
    if (
        (sk->joint[j1].x!=sk->joint2D[j1].x) ||
        (sk->joint[j1].y!=sk->joint2D[j1].y)
    )
    {
        return 1;
    }

    return 0;
}



static int skeletonEmpty(struct skeletonHuman * sk)
{
    unsigned int i=0 , emptyJoints=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        if ( skeletonEmpty3DJoint(sk,i) )
        {
            ++emptyJoints;
        }
    }

    if (emptyJoints==HUMAN_SKELETON_MIRRORED_PARTS)
    {
        fprintf(stderr,  "All skeleton joints are empty.. \n"   );
        return 1;
    }

    return 0;
}

static int skeletonSameJoints(unsigned int j1 , unsigned int j2)
{
    if (humanSkeletonJointsRelationMap[j1]==j2)
    {
        return 1;
    }
    return 0;
}

static void updateSkeletonBoundingBox(struct skeletonHuman * sk)
{
    //Use joints to extract bbox
    float minX = sk->joint[HUMAN_SKELETON_HEAD].x;
    float maxX = sk->joint[HUMAN_SKELETON_HEAD].x;
    float minY = sk->joint[HUMAN_SKELETON_HEAD].y;
    float maxY = sk->joint[HUMAN_SKELETON_HEAD].y;
    float minZ = sk->joint[HUMAN_SKELETON_HEAD].z;
    float maxZ = sk->joint[HUMAN_SKELETON_HEAD].z;

    unsigned int i=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        if (
            (sk->joint[i].x!=0.0) ||
            (sk->joint[i].y!=0.0) ||
            (sk->joint[i].z!=0.0)
        )
        {
            if (sk->joint[i].x>maxX)
            {
                maxX = sk->joint[i].x;
            }
            else if (sk->joint[i].x<minX)
            {
                minX = sk->joint[i].x;
            }

            if (sk->joint[i].y>maxY)
            {
                maxY = sk->joint[i].y;
            }
            else if (sk->joint[i].y<minY)
            {
                minY = sk->joint[i].y;
            }

            if (sk->joint[i].z>maxZ)
            {
                maxZ = sk->joint[i].z;
            }
            else if (sk->joint[i].z<minZ)
            {
                minZ = sk->joint[i].z;
            }
        }
    }

    sk->bbox[0].x = maxX;
    sk->bbox[0].y = maxY;
    sk->bbox[0].z = minZ;
    sk->bbox[1].x = maxX;
    sk->bbox[1].y = minY;
    sk->bbox[1].z = minZ;
    sk->bbox[2].x = minX;
    sk->bbox[2].y = minY;
    sk->bbox[2].z = minZ;
    sk->bbox[3].x = minX;
    sk->bbox[3].y = maxY;
    sk->bbox[3].z = minZ;
    sk->bbox[4].x = maxX;
    sk->bbox[4].y = maxY;
    sk->bbox[4].z = maxZ;
    sk->bbox[5].x = maxX;
    sk->bbox[5].y = minY;
    sk->bbox[5].z = maxZ;
    sk->bbox[6].x = minX;
    sk->bbox[6].y = minY;
    sk->bbox[6].z = maxZ;
    sk->bbox[7].x = minX;
    sk->bbox[7].y = maxY;
    sk->bbox[7].z = maxZ;

    sk->bboxDimensions.x = (float) sk->bbox[4].x-sk->bbox[2].x;
    sk->bboxDimensions.y = (float) sk->bbox[4].y-sk->bbox[2].y;
    sk->bboxDimensions.z = (float) sk->bbox[4].z-sk->bbox[2].z;
}






static double get3DAngleABC( double* srcPoint, double* dstPoint, double* defaultPoint )
{
    double *a = dstPoint, *b = srcPoint , *c = defaultPoint;
    /*
                          * dstPoint
                        /
                    /
                /
     srcPoint/   Result
          *  -  -  -  -  -  *  defaultPoint

    */
    double ab[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
    double bc[3] = { c[0] - b[0], c[1] - b[1], c[2] - b[2]  };

    double abVec = sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
    double bcVec = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);

    double abNorm[3] = {ab[0] / abVec, ab[1] / abVec, ab[2] / abVec};
    double bcNorm[3] = {bc[0] / bcVec, bc[1] / bcVec, bc[2] / bcVec};

    double res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] + abNorm[2] * bcNorm[2];

    return acos(res)*180.0/ 3.141592653589793;
}




static double get2DAngleABC( double* srcPoint, double* dstPoint, double* defaultPoint )
{
    double *a = dstPoint, *b = srcPoint , *c = defaultPoint;
    /*
                          * dstPoint
                        /
                    /
                /
     srcPoint/   Result
          *  -  -  -  -  -  *  defaultPoint

    */

    double ab[2] = { b[0] - a[0], b[1] - a[1] };
    double bc[2] = { c[0] - b[0], c[1] - b[1] };

    double abVec = sqrt(ab[0] * ab[0] + ab[1] * ab[1] );
    double bcVec = sqrt(bc[0] * bc[0] + bc[1] * bc[1] );

    double abNorm[2] = {ab[0] / abVec, ab[1] / abVec };
    double bcNorm[2] = {bc[0] / bcVec, bc[1] / bcVec };

    double res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] ;

    res = acos(res)*180.0/ 3.141592653589793;


    /*
     fprintf(stderr,"angle(%0.2f,%0.2f,%0.2f) = %0.2f  [ ab=%0.2f bc=%0.2f abVec=%0.2f bcVec=%0.2f abNorm=%0.2f bcNorm=%0.2f ] ",*srcPoint,*dstPoint,*defaultPoint,res
            ,ab,bc,abVec,bcVec,abNorm,bcNorm
             );
    */

    return res;
}



static double getAngleABCRelative( double* srcPointA , double* srcPointB ,
                                   double* dstPointA , double* dstPointB ,
                                   double* srcDefaultPointA ,  double* srcDefaultPointB ,
                                   double* dstDefaultPointA ,  double* dstDefaultPointB  )
{
    double relativeSrc[2] = { 0 , 0 };
    double relativeDst[2] = { *dstPointA - * srcPointA, *dstPointB - * srcPointB };
    double relativeDefaultDst[2] = { *dstDefaultPointA - * srcDefaultPointA , *dstDefaultPointB - * srcDefaultPointB };


    return get2DAngleABC(relativeSrc,relativeDst,relativeDefaultDst);
}


enum Point2FlatArray
{
    p_X = 0 ,
    p_Y ,
    p_Z
};


static void updateSkeletonAnglesGeneric(struct skeletonHuman * sk , float * defJoints)
{
    unsigned int i=0;
    unsigned int src=0,dst=0;

    double srcDA,srcDB , dstDA,dstDB , srcDefDA,srcDefDB , dstDefDA,dstDefDB;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        src = humanSkeletonJointsRelationMap[i];
        dst = i;

        if ( !skeletonSameJoints(src,dst) )
        {
           if (skeletonEmpty3DJoint( sk , src ) ) {  sk->active[i]=0; fprintf(stderr,"SRC Joint %s is empty \n" , jointNames[i]); } else
           if (skeletonEmpty3DJoint( sk , dst ) ) {  sk->active[i]=0; fprintf(stderr,"DST Joint %s is empty \n" , jointNames[i]); }
             else
           {
            sk->active[i]=1;
            //Z and Y gives X
            srcDA = (double) sk->joint[src].z;
            srcDB = (double) sk->joint[src].y;
            dstDA = (double) sk->joint[dst].z;
            dstDB = (double) sk->joint[dst].y;
            srcDefDA = (double) defaultJoints[src*3+p_Z];
            srcDefDB = (double) defaultJoints[src*3+p_Y];
            dstDefDA = (double) defaultJoints[dst*3+p_Z];
            dstDefDB = (double) defaultJoints[dst*3+p_Y];
            sk->relativeJointAngle[i].x=getAngleABCRelative(&srcDA,&srcDB,&dstDA,&dstDB,&srcDefDA,&srcDefDB,&dstDefDA,&dstDefDB);

            //Z and X gives Y
            srcDA = (double) sk->joint[src].z;
            srcDB = (double) sk->joint[src].x;
            dstDA = (double) sk->joint[dst].z;
            dstDB = (double) sk->joint[dst].x;
            srcDefDA = (double) defaultJoints[src*3+p_Z];
            srcDefDB = (double) defaultJoints[src*3+p_X];
            dstDefDA = (double) defaultJoints[dst*3+p_Z];
            dstDefDB = (double) defaultJoints[dst*3+p_X];
            sk->relativeJointAngle[i].y=getAngleABCRelative(&srcDA,&srcDB,&dstDA,&dstDB,&srcDefDA,&srcDefDB,&dstDefDA,&dstDefDB);

            //X and Y gives Z
            srcDA = (double) sk->joint[src].x;
            srcDB = (double) sk->joint[src].y;
            dstDA = (double) sk->joint[dst].x;
            dstDB = (double) sk->joint[dst].y;
            srcDefDA = (double) defaultJoints[src*3+p_X];
            srcDefDB = (double) defaultJoints[src*3+p_Y];
            dstDefDA = (double) defaultJoints[dst*3+p_X];
            dstDefDB = (double) defaultJoints[dst*3+p_Y];
            sk->relativeJointAngle[i].z=getAngleABCRelative(&srcDA,&srcDB,&dstDA,&dstDB,&srcDefDA,&srcDefDB,&dstDefDA,&dstDefDB);
           }
        }


      // sk->active[i]=0;
      // if ( (sk->relativeJointAngle[i].x!=0.0) && ( sk->relativeJointAngle[i].x==sk->relativeJointAngle[i].x ) ) {  sk->active[i]=1; }
      // if ( (sk->relativeJointAngle[i].y!=0.0) && ( sk->relativeJointAngle[i].y==sk->relativeJointAngle[i].y ) ) {  sk->active[i]=1; }
      // if ( (sk->relativeJointAngle[i].z!=0.0) && ( sk->relativeJointAngle[i].z==sk->relativeJointAngle[i].z ) ) {  sk->active[i]=1; }
    }


    if ( !sk->active[HUMAN_SKELETON_LEFT_KNEE] )  { sk->active[HUMAN_SKELETON_LEFT_HIP]=0; }
    if ( !sk->active[HUMAN_SKELETON_RIGHT_KNEE] ) { sk->active[HUMAN_SKELETON_RIGHT_HIP]=0; }

}



static void updateSkeletonAngles(struct skeletonHuman * sk)
{
    updateSkeletonAnglesGeneric( sk , defaultJoints);
}

static void updateSkeletonAnglesNAO(struct skeletonHuman * sk)
{
    updateSkeletonAnglesGeneric( sk , NAOdefaultJoints);
}


static int convertSkeletonFlat3DJointsToPoint3D(struct point3D * out  , float * in )
{
    unsigned int i=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        out[i].x = in[i*3+p_X];
        out[i].y = in[i*3+p_Y];
        out[i].z = in[i*3+p_Z];

    }
    return 1;
}



static int convertSkeletonFlat2DJointsToPoint2D(struct point2D * out  , float * in )
{
    unsigned int i=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        out[i].x = in[i*2+p_X];
        out[i].y = in[i*2+p_Y];

    }
    return 1;
}


static int fillWithDefaultSkeleton(struct skeletonHuman * sk)
{
    convertSkeletonFlat3DJointsToPoint3D( sk->joint  , defaultJoints );
    convertSkeletonFlat2DJointsToPoint2D( sk->joint2D , defaultJoints2D );
    updateSkeletonAngles(sk);
    return 1;
}


static int fillWithDefaultNAOSkeleton(struct skeletonHuman * sk)
{
    convertSkeletonFlat3DJointsToPoint3D( sk->joint  , NAOdefaultJoints );
    convertSkeletonFlat2DJointsToPoint2D( sk->joint2D , NAOdefaultJoints2D );
    updateSkeletonAngles(sk);
    return 1;
}



static int visualizeSkeletonHuman(const char * filename , struct skeletonHuman * sk , float scale)
{
    unsigned int origX=640,origY=480;
    float origXHalf=(float) origX/2,origYHalf=(float) origY/2;


    unsigned int i=0;
    unsigned int src=0,dst=0;
    FILE * fp = fopen(filename,"w");
    if (fp!=0)
    {
        fprintf(fp,"<svg width=\"%0.0f\" height=\"%0.0f\">\n" , origX*scale , origY*scale);

        fprintf(fp,"<text x=\"0\" y=\"20\">\n Timestamp %u\n</text>\n",sk->observationNumber);

        fprintf(fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,255,0);stroke-width:2;stroke-dasharray:10,10\" />\n",
                (float) origXHalf*scale, 0.0 , (float)origXHalf*scale, (float) origY*scale );

        fprintf(fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,255,0);stroke-width:2;stroke-dasharray:10,10\" />\n",
                0.0 ,(float) origYHalf*scale , (float) origX*scale  , (float)origYHalf*scale );


        for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
            src = humanSkeletonJointsRelationMap[i];
            dst = i;

            if ( ( !skeletonSameJoints(src,dst) ) && (!skeletonEmpty2DJoint(sk,src)) && (!skeletonEmpty2DJoint(sk,dst)) )
            {
                fprintf(fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,0,0);stroke-width:2\" />\n",
                        sk->joint2D[src].x*scale, sk->joint2D[src].y*scale , sk->joint2D[dst].x*scale, sk->joint2D[dst].y*scale );
            }
        }

        for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
            if (!skeletonEmpty2DJoint(sk,i))
            {
                fprintf(fp,"<circle cx=\"%0.2f\" cy=\"%0.2f\" r=\"10\" stroke=\"green\" stroke-width=\"4\" fill=\"yellow\" />\n", sk->joint2D[i].x*scale, sk->joint2D[i].y*scale );


                fprintf(fp,"<text x=\"%0.2f\" y=\"%0.2f\">\n",(sk->joint2D[i].x*scale)-40, (sk->joint2D[i].y*scale)-60   );
                fprintf(fp,"  <tspan fill=\"red\">%s</tspan>",jointNames[i]);
                fprintf(fp,"</text>\n"   );

                fprintf(fp,"<text x=\"%0.2f\" y=\"%0.2f\">\n",(sk->joint2D[i].x*scale)-40, (sk->joint2D[i].y*scale)-40   );
                fprintf(fp,"  <tspan fill=\"red\">%0.2f</tspan>,",sk->joint[i].x);
                fprintf(fp,"  <tspan fill=\"green\">%0.2f</tspan>,",sk->joint[i].y);
                fprintf(fp,"  <tspan fill=\"blue\">%0.2f</tspan>\n",sk->joint[i].z);
                fprintf(fp,"</text>\n"   );

                fprintf(fp,"<text x=\"%0.2f\" y=\"%0.2f\">\n",(sk->joint2D[i].x*scale)-40, (sk->joint2D[i].y*scale)-20   );
                fprintf(fp,"  <tspan fill=\"red\">%0.2f</tspan>/",sk->relativeJointAngle[i].x);
                fprintf(fp,"  <tspan fill=\"green\">%0.2f</tspan>/",sk->relativeJointAngle[i].y);
                fprintf(fp,"  <tspan fill=\"blue\">%0.2f</tspan>\n",sk->relativeJointAngle[i].z);
                fprintf(fp,"</text>\n"   );
            }

        }


        fprintf(fp,"</svg>\n");

        fclose(fp);
        return 1;
    }
    return 0;
}



#ifdef __cplusplus
}
#endif

#endif // SKELETON_H_INCLUDED

