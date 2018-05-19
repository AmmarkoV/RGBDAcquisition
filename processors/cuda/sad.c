//https://sites.google.com/a/nirmauni.ac.in/cudacodes/ongoing-projects/automatic-conversion-of-source-code-for-c-to-cuda-c/converted-programs/sum-of-absolute-differences
/*Parth C2CUDA Generated CODE */
#include <stdio.h> 
#include <stdlib.h> 
#include <cuda_runtime.h>
#include <string.h>
#define ROWSIZE 4
#define COLSIZE 4

__global__ void kernel( int *gpu_sum, int **gpu_a , size_t pitch2, int **gpu_b , size_t pitch3, int **gpu_c , size_t pitch4,int *gpu_c2) {
     int idx=threadIdx.x+blockIdx.x*blockDim.x;
      int idy=threadIdx.y+blockIdx.y*blockDim.y;

     //Pitch Slices per Row for 2D Array
     int* rowa = (int*)((char*)gpu_a + idx*pitch2);
     int* rowb = (int*)((char*)gpu_b + idx*pitch3);
     int* rowc = (int*)((char*)gpu_c + idx*pitch4);

      int gpu_i =idx;
       
          for (int gpu_j = 0; gpu_j < COLSIZE; gpu_j++) {
              int* rowbi = (int* ) ((char* )gpu_b + (gpu_i) * pitch3);
              int* rowai = (int* ) ((char* )gpu_a + (gpu_i) * pitch2);
              int* rowci = (int* ) ((char* )gpu_c + (gpu_i) * pitch4);
              rowci[gpu_j] = rowbi[gpu_j] - rowai[gpu_j];
          }
             __syncthreads();
               for (int gpu_j = 0; gpu_j < COLSIZE; gpu_j++) {
              gpu_c2[gpu_i] += rowc[gpu_j];
              }
            __syncthreads();
             for (gpu_i = 0; gpu_i < ROWSIZE; gpu_i++) {
         *gpu_sum += gpu_c2[gpu_i];
             }
    __syncthreads();
}
int main(int argc, char *argv[]) {
        int i;
        int j;

        //Kernel Variables
        int sum=0;int *g_sum;
        int a[ROWSIZE][COLSIZE];int **g_a;
        int b[ROWSIZE][COLSIZE];int **g_b;
        int c[ROWSIZE][COLSIZE];int **g_c;
        int c2[ROWSIZE];int *g_c2;

        //CUDA GRID BLOCK SIZE AND NUMBER OF BLOCKS
        int block_size = 4;
        const int N = 1;//ROWSIZE * ROWSIZE * COLSIZE * ROWSIZE * COLSIZE;  // Number of elements in arrays
        int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);

        for (i = 0; i < ROWSIZE; i++) {
             for (j = 0; j < COLSIZE; j++) {
                  a[i][j] = 1;
                  b[i][j] = 2;
                  c[i][j] = 0;
                  c2[i] = 0;
             }
        }

        // Memory Allocation
        g_sum= (int *)malloc(sizeof(int)); // Allocate variable on host
         cudaMalloc((void **) &g_sum,sizeof(int)); // Allocate variable on device
        size_t pitch2;
         cudaMallocPitch(&g_a, &pitch2, COLSIZE * sizeof(int), ROWSIZE);
         // Allocate 2Darray on device
        size_t pitch3;
         cudaMallocPitch(&g_b, &pitch3, COLSIZE * sizeof(int), ROWSIZE);
         // Allocate 2Darray on device
        size_t pitch4;
         cudaMallocPitch(&g_c, &pitch4, COLSIZE * sizeof(int), ROWSIZE);
         // Allocate 2Darray on device
        g_c2= (int *)malloc(ROWSIZE *sizeof(int)); // Allocate array on host
         cudaMalloc((void **) &g_c2,ROWSIZE * sizeof(int)); // Allocate array on device

        // Copy Data to device from host
        cudaMemcpy(g_sum, &sum, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy2D(g_a, pitch2, a, COLSIZE * sizeof(int), COLSIZE * sizeof(int),ROWSIZE, cudaMemcpyHostToDevice);
        cudaMemcpy2D(g_b, pitch3, b, COLSIZE * sizeof(int), COLSIZE * sizeof(int),ROWSIZE, cudaMemcpyHostToDevice);
        cudaMemcpy2D(g_c, pitch4, c, COLSIZE * sizeof(int), COLSIZE * sizeof(int),ROWSIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(g_c2, c2,ROWSIZE * sizeof(int), cudaMemcpyHostToDevice);

        // call kernel
         kernel <<< n_blocks, block_size >>>( g_sum, g_a,pitch2, g_b,pitch3, g_c,pitch4, g_c2);

        // Retrieve result from device and store it in host array
        cudaMemcpy(&sum, g_sum, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy2D(a,COLSIZE * sizeof(int), g_a,pitch2,COLSIZE * sizeof(int),ROWSIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(b,COLSIZE * sizeof(int), g_b,pitch3,COLSIZE * sizeof(int),ROWSIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(c,COLSIZE * sizeof(int), g_c,pitch4,COLSIZE * sizeof(int),ROWSIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(c2, g_c2,ROWSIZE * sizeof(int), cudaMemcpyDeviceToHost);

        // Free GPU Variables
        cudaFree(g_sum);
        cudaFree(g_a);
        cudaFree(g_b);
        cudaFree(g_c);
        cudaFree(g_c2);


        printf("\n Sum = %d",sum);
        printf("\n  blocksize=%d numofblock=%d\n",block_size,n_blocks);
         system("pause");
         return 0;

}
//// Pattern :
////__global__ void kernel(parameters) {
////    forloop(gpu_j < COLSIZE) {
////         pitchstat
//         pitchstat
//         pitchstat
//         assignStat
////    }
//    assignStat
//    assignStat
////}

