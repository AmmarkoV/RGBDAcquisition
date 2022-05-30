

#include "openCLRenderer.h"
#include "openCLTools.h"

#include <stdio.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <fcntl.h>


#define BUF_SIZE 2048
#define ERROR -1

#define END 0

char *matrixMultiplicationSource = "";
char *transform3DPointSource     = "";


void printMatrix(float *arr, int n, int m)
{
    int i, j;
    for (i = 0 ; i < n ; ++i )
        {
            for (j = 0 ; j < m ; ++j )
                {
                    printf("%0.2f ", arr[i*n+j]);
                }
            printf("\n" );
        }
    printf("\n" );
}






void transform3DPointHost(
                          unsigned int numberOf3DPoints,
                          float * transformation4x4,
                          float * input3DPoints,
                          int *   indices,
                          float * result3DPoints
                         )
{
  for (int idx = 0;  idx<numberOf3DPoints; idx++)
  {
  //-------------------------------------------------------------------------------
  float * m = &transformation4x4[idx*16];
  float X=input3DPoints[idx*4+0];
  float Y=input3DPoints[idx*4+1];
  float Z=input3DPoints[idx*4+2];
  float W=input3DPoints[idx*4+3];
  //-------------------------------------------------------------------------------  
  float * resultPoint3D = &result3DPoints[idx*4];
  //-------------------------------------------------------------------------------
  resultPoint3D[0] = m[3]  * W + m[0]  * X + m[1]  * Y + m[2]  * Z;
  resultPoint3D[1] = m[7]  * W + m[4]  * X + m[5]  * Y + m[6]  * Z;
  resultPoint3D[2] = m[11] * W + m[8]  * X + m[9]  * Y + m[10] * Z;
  resultPoint3D[3] = m[15] * W + m[12] * X + m[13] * Y + m[14] * Z;
  //-------------------------------------------------------------------------------

  // Ok we have our results but now to normalize our vector
  if (resultPoint3D[3]!=0.0)
  {
   resultPoint3D[0]/=resultPoint3D[3];
   resultPoint3D[1]/=resultPoint3D[3];
   resultPoint3D[2]/=resultPoint3D[3];
   resultPoint3D[3]=1.0; // resultPoint3D[3]/=resultPoint3D[3];
  }    
  }
}


void randomizeFloatArray(float * arr,unsigned int numberOfItems)
{
  for (int i=0; i<numberOfItems; i++)
  {
      arr[i]=rand();
  }
}



int oclr_initialize(struct oclRenderer * or)
{
    
    
}




int main(int argc, char const *argv[])
{
    //-----------------------------------------------------------

    //-----------------------------------------------------------


    
    cl_platform_id platform_id;
    cl_uint num_of_platforms;
    cl_device_id device_id;
    cl_uint num_of_devices;
    cl_context context;
    cl_context_properties properties[3]; 
    cl_command_queue command_queue;
    cl_int err;
    size_t local[2], global[2];
    int len;
    int res;
    int n = 3;
    int inp_len = sizeof(int)*n*n, op_len = sizeof(int)*n*n;
    float matA[] = {1,2,5,4,2,5,5,2,6}, matB[] = {5,2,7,3,6,2,6,2,6};
    float matC[9];

    printMatrix(matA,n,n);
    printMatrix(matB,n,n);

    matrixMultiplicationSource = read_file("matrixMultiplication.cl",&len);
    transform3DPointSource     = read_file("transform3DPoint.cl",&len);

    if (clGetPlatformIDs(1,&platform_id,&num_of_platforms) != CL_SUCCESS)
        {
            printf("Platform ID Error\n");
            return -1;
        }

    if (clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_ALL,1,&device_id,&num_of_devices))
        {
            printf("Device ID error\n");
            return -1;
        }
        
        
     cl_device_id *devices; 
     cl_uint addr_data;
     char name_data[48], ext_data[4096];

     devices = (cl_device_id*)  malloc(sizeof(cl_device_id) * num_of_devices);
     if (devices!=0)
     {
       clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL,num_of_devices, devices, NULL);
       printf("%u OpenCL devices\n",num_of_devices);
     
       for(int i=0; i<num_of_devices; i++) { 
                                             int err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name_data), name_data, NULL);
                                             if(err < 0) {
                                                           perror("Couldn't read extension data");
                                                           return -1;
                                                         }
                                             clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS,sizeof(ext_data), &addr_data, NULL);
                                             clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS,sizeof(ext_data), ext_data, NULL);
                                             printf("NAME: %s\nADDRESS_WIDTH: %u\nEXTENSIONS: %s\n",name_data, addr_data, ext_data);
                                           }
      free(devices); 
     }


    //-------------------------------------------------------------------------------------------------------------
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties) platform_id;
    properties[2] = 0;
    //-------------------------------------------------------------------------------------------------------------
    context = clCreateContext(properties,1,&device_id,NULL,NULL,&err);    checkOpenCLError(err,__FILE__, __LINE__);
    //-------------------------------------------------------------------------------------------------------------
    command_queue = clCreateCommandQueue(context,device_id,0,&err);       checkOpenCLError(err,__FILE__, __LINE__);
    //-------------------------------------------------------------------------------------------------------------
    cl_mem inputA = clCreateBuffer(context,CL_MEM_READ_ONLY,inp_len,NULL,&err);  checkOpenCLError(err,__FILE__, __LINE__);
    cl_mem inputB = clCreateBuffer(context,CL_MEM_READ_ONLY,inp_len,NULL,&err);  checkOpenCLError(err,__FILE__, __LINE__);
    cl_mem output = clCreateBuffer(context,CL_MEM_WRITE_ONLY,op_len,NULL,&err);  checkOpenCLError(err,__FILE__, __LINE__);
    //-------------------------------------------------------------------------------------------------------------
    clEnqueueWriteBuffer(command_queue,inputA,CL_TRUE,0,inp_len,matA,0,NULL,NULL);
    clEnqueueWriteBuffer(command_queue,inputB,CL_TRUE,0,inp_len,matB,0,NULL,NULL);
    //-------------------------------------------------------------------------------------------------------------


    
    cl_program programMatrixMultiplication = clCreateProgramWithSource(context,1,(const char**) &matrixMultiplicationSource, NULL, &err); checkOpenCLError(err,__FILE__, __LINE__);
    if (res = clBuildProgram(programMatrixMultiplication,1,&device_id,NULL,NULL,NULL) != CL_SUCCESS)
        {
            printf("Error building program: %s\n",getErrorString(res));
            return -1;
        }
    cl_kernel kernel = clCreateKernel(programMatrixMultiplication,"matrixMultiplication",&err); checkOpenCLError(err,__FILE__, __LINE__);
    if  ( 
          (clSetKernelArg(kernel,0,sizeof(cl_mem),&inputA) != CL_SUCCESS) ||
          (clSetKernelArg(kernel,1,sizeof(cl_mem),&inputB) != CL_SUCCESS) ||
          (clSetKernelArg(kernel,2,sizeof(cl_mem),&output) != CL_SUCCESS)
        )
        {
            printf("Kernel setting error\n");
        }

    //------------
    global[0] = 3;
    global[1] = 3;
    //------------
    if  ( 
          (clEnqueueNDRangeKernel(command_queue,kernel,2,NULL,global,NULL,0,NULL,NULL) != CL_SUCCESS ) ||
          (clEnqueueReadBuffer(command_queue,output,CL_TRUE,0,op_len,matC,0,NULL,NULL) != CL_SUCCESS )
        )
        {
            printf("Buffer read error\n");
            return -1;
        }
    printMatrix(matC,n,n);





    #define NUMBER_OF_POINTS 10024
    float transformationMatrices4x4[16*NUMBER_OF_POINTS]={0};
    float input3DPoints[4 *NUMBER_OF_POINTS]={0};
    int indices[16]={
                       3 ,0 ,1 ,2,
                       7 ,4 ,5 ,6,
                       11,8 ,9 ,10,
                       15,12,13,14
                    };
    float output3DPoints[4*NUMBER_OF_POINTS]={0};
    float output3DHostPoints[4*NUMBER_OF_POINTS]={0};
   
    randomizeFloatArray(transformationMatrices4x4,16*NUMBER_OF_POINTS);
    randomizeFloatArray(input3DPoints,4*NUMBER_OF_POINTS);

    unsigned long startTime = GetTickCountMicrosecondsOCL(); 
    transform3DPointHost(
                          NUMBER_OF_POINTS,
                          transformationMatrices4x4,
                          input3DPoints,
                          indices,
                          output3DHostPoints
                         );
    unsigned long endTime = GetTickCountMicrosecondsOCL(); 
    fprintf(stderr,"Host Point 3D Transform = %lu microseconds\n",endTime-startTime);


    //---------------------------------------------------------------------------------------------------------------------------------------------------------------
    cl_mem inputTransformation4x4 = clCreateBuffer(context,CL_MEM_READ_ONLY ,16*NUMBER_OF_POINTS*sizeof(float),NULL,&err);   checkOpenCLError(err,__FILE__, __LINE__);
    cl_mem inputinput3DPoints     = clCreateBuffer(context,CL_MEM_READ_ONLY ,4 *NUMBER_OF_POINTS*sizeof(float),NULL,&err);   checkOpenCLError(err,__FILE__, __LINE__);
    cl_mem inputIndices           = clCreateBuffer(context,CL_MEM_READ_ONLY ,16*sizeof(int),NULL,&err);                    checkOpenCLError(err,__FILE__, __LINE__);
    cl_mem outputresult3DPoints   = clCreateBuffer(context,CL_MEM_WRITE_ONLY,4*NUMBER_OF_POINTS*sizeof(float),NULL,&err);   checkOpenCLError(err,__FILE__, __LINE__);
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------
    clEnqueueWriteBuffer(command_queue,inputTransformation4x4,CL_TRUE,0,16*NUMBER_OF_POINTS*sizeof(float),transformationMatrices4x4,0,NULL,NULL);
    clEnqueueWriteBuffer(command_queue,inputinput3DPoints,    CL_TRUE,0,4 *NUMBER_OF_POINTS*sizeof(float),input3DPoints,0,NULL,NULL);
    clEnqueueWriteBuffer(command_queue,inputIndices,          CL_TRUE,0,16*sizeof(int),indices,0,NULL,NULL);
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------


    
    cl_program programTransform3DPoint = clCreateProgramWithSource(context,1,(const char**) &transform3DPointSource, NULL, &err); checkOpenCLError(err,__FILE__, __LINE__);
    if (res = clBuildProgram(programTransform3DPoint,1,&device_id,NULL,NULL,NULL) != CL_SUCCESS)
        {
            printf("Error building program: %s\n",getErrorString(res));
            getBuildError(programTransform3DPoint,&device_id);
            return -1;
        }
    cl_kernel kernelTransform3DPoint = clCreateKernel(programTransform3DPoint,"transform3DPoint",&err); checkOpenCLError(err,__FILE__, __LINE__);
    if  ( 
          (clSetKernelArg(kernelTransform3DPoint,0,sizeof(cl_mem),&inputTransformation4x4) != CL_SUCCESS) ||
          (clSetKernelArg(kernelTransform3DPoint,1,sizeof(cl_mem),&inputinput3DPoints)     != CL_SUCCESS) ||
          (clSetKernelArg(kernelTransform3DPoint,2,sizeof(cl_mem),&inputIndices)           != CL_SUCCESS) ||
          (clSetKernelArg(kernelTransform3DPoint,3,sizeof(cl_mem),&outputresult3DPoints)   != CL_SUCCESS)
        )
        {
            printf("Kernel setting error\n");
        }

    //------------
    global[0] = NUMBER_OF_POINTS;
    global[1] = 4; //X,Y,Z,W
    startTime = GetTickCountMicrosecondsOCL(); 
    //------------
    if  ( 
          (clEnqueueNDRangeKernel(command_queue,kernelTransform3DPoint,2,NULL,global,NULL,0,NULL,NULL) != CL_SUCCESS ) 
        )
        {
            printf("Could not execute  error\n");
            return -1;
        }
    endTime = GetTickCountMicrosecondsOCL(); 
    fprintf(stderr,"OpenCL Point 3D Transform = %lu microseconds\n",endTime-startTime);


   if  (clEnqueueReadBuffer(command_queue,outputresult3DPoints,CL_TRUE,0,4*NUMBER_OF_POINTS*sizeof(float),output3DPoints,0,NULL,NULL) != CL_SUCCESS )
        {
            printf("Buffer read error\n");
            return -1;
        }

    for (int i=0; i<NUMBER_OF_POINTS; i++)
    {
     float hW = output3DHostPoints[i*4+3];
     if (hW==0.0) { hW=1.0; }
     float hX = output3DHostPoints[i*4+0]/hW;
     float hY = output3DHostPoints[i*4+1]/hW;
     float hZ = output3DHostPoints[i*4+2]/hW;
     
     float dW = output3DPoints[i*4+3];
     if (dW==0.0) { dW=1.0; }
     float dX = output3DPoints[i*4+0]/dW;
     float dY = output3DPoints[i*4+1]/dW;
     float dZ = output3DPoints[i*4+2]/dW;

     if ( (hX!=dX) || (hY!=dY) || (hZ!=dZ) ) 
     {
       fprintf(stderr,"Difference in Point %u  = ",i);
       fprintf(stderr,"%0.2f,%0.2f,%0.2f  ",hX,hY,hZ);
       fprintf(stderr,"%0.2f,%0.2f,%0.2f\n",dX,dY,dZ);
     } 
    }
     


    return 0;
}
