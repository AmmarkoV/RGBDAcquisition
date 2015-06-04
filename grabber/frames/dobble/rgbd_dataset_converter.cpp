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

#define LABEL_LIST "label.list"
#define TMP_ORDERED_LIST "tmp.list"
#define TMP_SHUFFLED_LIST "tmpR.list"
#define CAFFEDIR "/home/ammar/Documents/Programming/FORTH/input_acquisition/3dparty/caffe"
#define STRSZ 2048


using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

int doShuffle=1;

int execute(char * cmd)
{
 return ( system(cmd) == 0 ) ;
}

int moveFile(char * filenameSrc,char * filenameDst)
{
 char commandline[STRSZ]={0};
 snprintf(commandline,STRSZ,"mv %s %s",filenameSrc,filenameDst);
 return execute(commandline);
}


int deleteFile(char * filename)
{
 fprintf(stderr,"Erasing file %s ",filename);
 char commandline[STRSZ]={0};
 snprintf(commandline,STRSZ,"rm %s",filename);

 return execute(commandline);
}


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


int appendFileList(char * filename,char * className,unsigned int label)
{
  FILE * pFile = fopen (filename, "a");
  if (pFile==0) { return 0; }

  fprintf(pFile,"%u %s\n",label,className);
  fclose(pFile);
  return 1;
}


int deleteFolder(char * dir)
{
 fprintf(stderr,"Erasing dir %s ",dir);
 char commandline[STRSZ]={0};
 snprintf(commandline,STRSZ,"rm -rf %s",dir);

 return execute(commandline);
}



int generate_mean(char * caffedir, char * outputdir)
{
 fprintf(stderr,"generate_mean using %s as caffedir\n",caffedir);
 char commandline[STRSZ]={0};
 snprintf(commandline,STRSZ,"%s/build/tools/compute_image_mean %s/../leveldb %s/../leveldb/mean.binaryproto",caffedir,outputdir,outputdir);

 return execute(commandline);
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

int shuffle_filelist(char * filein,char * fileout)
{
 char commandline[STRSZ]={0};
 snprintf(commandline,STRSZ,"shuf %s > %s",filein,fileout);
 execute(commandline);

 snprintf(commandline,STRSZ,"rm %s",filein);
 execute(commandline);

 snprintf(commandline,STRSZ,"mv %s %s",fileout,filein);
 execute(commandline);

 return 1;
}




void generateFileList(const char * inputdir)
{
unsigned int label=0;
char datasetFilename[STRSZ]={0};
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
      snprintf(datasetFilename,STRSZ,"%s/%s",inputdir,classDirP->d_name);
      DIR *classFile = opendir(datasetFilename);
      if (classFile !=0)
      {
        appendFile(LABEL_LIST,classDirP->d_name);

        while ((classFileP=readdir(classFile)) != 0)
         {
          if ( (strcmp(classFileP->d_name,".")!=0) && (strcmp(classFileP->d_name,"..")!=0) )
          {
              snprintf(datasetFilename,STRSZ,"%s/%s/%s",inputdir,classDirP->d_name,classFileP->d_name);
              appendFileList(TMP_ORDERED_LIST,datasetFilename,label);
         }
        }
        ++label;
        closedir(classFile);
      }
    }
  }
  closedir(classDir);
}

 if (doShuffle) {  shuffle_filelist(TMP_ORDERED_LIST,TMP_SHUFFLED_LIST); }
}

void importFileListToLevelDB(const char * inputdir, const char * db_type)
{
    unsigned int width,height,channels,filesize;


    char datasetFilename[STRSZ]={0};
    snprintf(datasetFilename,STRSZ,"%s/../leveldb",inputdir);



    deleteFolder(datasetFilename);

    scoped_ptr<db::DB> train_db(db::GetDB(db_type));
    train_db->Open(datasetFilename, db::NEW);
    scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
    // Data buffer
    int label=0;
    Datum datum;


FILE * fp=fopen(TMP_ORDERED_LIST,"r");



char line[STRSZ]={0};
if (fp!=0)
{
 while (!feof(fp))
  {
   //We get a new line out of the file
   int readOpResult = (fgets(line,STRSZ,fp)!=0);
   if ( readOpResult != 0 )
    {
     char *pos;
     if ((pos=strchr(line, '\n')) != NULL) { *pos = '\0'; }


      char * endOfLabel = strstr(line," ");
      *endOfLabel=0;
      unsigned int label=atoi(line);
      endOfLabel=endOfLabel+1;


      //fprintf(stderr,"%s\n",endOfLabel);
      char * newImage = readImage(endOfLabel,&width,&height,&channels,&filesize);
      if (newImage!=0)
              {
               fprintf(stderr,"@%u  %ux%u:%u ",label,width,height,channels);
               datum.set_channels(channels);
               datum.set_height(height);
               datum.set_width(width);
               datum.set_label(label);
               datum.set_data(newImage, filesize);

               string out;
               CHECK(datum.SerializeToString(&out));
               txn->Put(string(endOfLabel,strlen(endOfLabel)), out);
               free(newImage);
               fprintf(stderr,"ok \n");
              } else
              {
                fprintf(stderr,"failed \n");
              }


    } //End of getting a line while reading the file
  }
 fclose(fp);
}
//-------------------------

 txn->Commit();
 train_db->Close();


 snprintf(datasetFilename,STRSZ,"%s/../leveldb/labels.nfo",inputdir);
 moveFile(LABEL_LIST,datasetFilename);

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

        //convert_dataset(argv[1],dbType);

        generateFileList(argv[1]);
        importFileListToLevelDB(argv[1],dbType);


        generate_mean(CAFFEDIR,argv[1]);
    }
    return 0;
}


