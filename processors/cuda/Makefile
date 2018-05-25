all: program

program: cudacode.o
    g++ -o program -L/usr/local/cuda/lib64 -lcuda -lcudart sad.c  cudacode.o 

cudacode.o:
    nvcc -c -arch=sm_20 cudacode.cu 

clean: rm -rf *o program
