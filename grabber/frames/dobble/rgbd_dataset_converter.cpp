//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"


#include <sys/stat.h>
#include <errno.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"


using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;



int clearFile(char * filename)
{
  FILE * pFile = fopen (filename, "w");
  if (pFile==0) { return 0; }
  fclose(pFile);
  return 1;
}


int appendFile(char * filename,char * className)
{
  FILE * pFile = fopen (filename, "a");
  if (pFile==0) { return 0; }

  fprintf(pFile,"%s\n",className);
  fclose(pFile);
  return 1;
}



char * readImage(char * filename,unsigned int * width, unsigned int * height , unsigned int * channels, unsigned int * filesize)
{
  char * outputBuffer = 0;
  cv::Mat img = cv::imread(filename, -1);
  if (img.data!=0)
  {
   cv::Size s = img.size();

   *channels=img.channels();
   *height=s.height;
   *width=s.width;
   *filesize = (*channels) * (*height) * (*width);

    outputBuffer = (char *) malloc( (*filesize) * sizeof(char) );

    if (outputBuffer!=0)
    {
     memcpy(outputBuffer,img.data,(*filesize));
    }

  }

  return outputBuffer;
}

void convert_dataset(const char * inputdir, const char * db_type)
{
    unsigned int width;
    unsigned int height;
    unsigned int channels;
    unsigned int filesize;


    char datasetLabelFilename[512]={0};
    snprintf(datasetLabelFilename,512,"%s/../leveldb/labels.nfo",inputdir);
    char datasetFilename[512]={0};
    snprintf(datasetFilename,512,"%s/../leveldb",inputdir);

    scoped_ptr<db::DB> train_db(db::GetDB(db_type));
    train_db->Open(datasetFilename, db::NEW);
    scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
    // Data buffer
    int label=0;
    Datum datum;


    clearFile(datasetLabelFilename);





//struct stat st;
struct dirent *classDirP={0};
struct dirent *classFileP={0};
// enter existing path to directory below
DIR *classDir = opendir(inputdir);
if (classDir !=0)
{
 while ((classDirP=readdir(classDir)) != 0)
  {
    if ( (strcmp(classDirP->d_name,".")!=0) && (strcmp(classDirP->d_name,"..")!=0) )
    {


      snprintf(datasetFilename,512,"%s/%s",inputdir,classDirP->d_name);
      DIR *classFile = opendir(datasetFilename);
      if (classFile !=0)
      {
        appendFile(datasetLabelFilename,classDirP->d_name);
        while ((classFileP=readdir(classFile)) != 0)
         {
          if ( (strcmp(classFileP->d_name,".")!=0) && (strcmp(classFileP->d_name,"..")!=0) )
          {
              snprintf(datasetFilename,512,"%s/%s/%s",inputdir,classDirP->d_name,classFileP->d_name);
              fprintf(stderr,"%s/%s - ",classDirP->d_name,classFileP->d_name);

              char * newImage = readImage(datasetFilename,&width,&height,&channels,&filesize);
              if (newImage!=0)
              {
               fprintf(stderr,"#%u %ux%u:%u ",label,width,height,channels);
               datum.set_channels(channels);
               datum.set_height(height);
               datum.set_width(width);
               datum.set_label(label);
               datum.set_data(newImage, filesize);

               string out;
               CHECK(datum.SerializeToString(&out));
               txn->Put(string(datasetFilename,strlen(datasetFilename)), out);
               free(newImage);
               fprintf(stderr,"ok \n");
              } else
              {
                fprintf(stderr,"failed \n");
              }
         }
        }
        ++label;
        closedir(classFile);
      }

    }
  }
  closedir(classDir);
}

 txn->Commit();
 train_db->Close();
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Please give the directory path to convert files to leveldb.\n");
    }
    else
    {
        google::InitGoogleLogging(argv[0]);

        char dbType[120]="lmdb";
        convert_dataset(argv[1],dbType);
    }
    return 0;
}


