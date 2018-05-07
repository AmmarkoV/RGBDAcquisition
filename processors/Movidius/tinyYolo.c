//This code is taken from the original Yolo->Caffe port
//https://github.com/TLESORT/YOLO-TensorRT-GIE-/blob/master/YOLODraw.cpp

#include "tinyYolo.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../tools/Drawing/drawing.h"

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 448;
static const int INPUT_W = 448;
static const int OUTPUT_SIZE = 1470;
const int BATCH_SIZE=1;

#define NUM_CLASSES 20
#define NUM_CELLS 7
#define NUM_TOP_CLASSES 2


float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

struct box
{
    float x, y, w, h;
};


struct sortable_bbox
{
    int index;
    int idx_class;
    float **probs;
};


float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}


int nms_comparator(const void *pa, const void *pb)
{
    struct sortable_bbox *a = (struct sortable_bbox *)pa;
    struct sortable_bbox *b = (struct sortable_bbox *)pb;
    float diff = a->probs[a->index][b->idx_class] - b->probs[b->index][b->idx_class];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(struct box *a, struct box *b)
{
    float w = overlap(a->x, a->w, b->x, b->w);
    float h = overlap(a->y, a->h, b->y, b->h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(struct box *a, struct box *b)
{
    float i = box_intersection(a, b);
    float u = a->w*a->h + b->w*b->h - i;
    return u;
}

float box_iou(struct box *a, struct box *b)
{
    return (float) box_intersection(a, b)/box_union(a, b);
}




//https://github.com/TLESORT/YOLO-TensorRT-GIE-/blob/master/YOLODraw.cpp
void draw_detections(
                     char * pixels, unsigned int imageWidth, unsigned int imageHeight  ,
                     int num, float thresh, struct box *boxes, float **probs, const char **names)
{
    int i;
    for(i = 0; i < num; ++i)
     {
		float max = -1e10; int idx_class = -1;

		//Find biggest probability for i
		for (int j = 0; j < NUM_CLASSES; ++j)
		{
			if (probs[i][j] > max)
			{
				max = probs[i][j];
				idx_class = j;
			}
		}

        float prob = 0.0;
        if (idx_class!=-1) { prob = probs[i][idx_class]; }
        if(prob > thresh)
         {
            //int width = pow(prob, 1./2.)*10+1;

            int offset = idx_class*17 % NUM_CLASSES;
            float red   = get_color(0,offset,NUM_CLASSES) * 255;
            float green = get_color(1,offset,NUM_CLASSES) * 255;
            float blue  = get_color(2,offset,NUM_CLASSES) * 255;
            struct box b = boxes[i];

            float halfW = (float) b.w/2.;
            float halfH = (float) b.h/2.;


            float left  = (float) (b.x-halfW);
            float right = (float) (b.x+halfW);
            float top   = (float) (b.y-halfH);
            float bot   = (float) (b.y+halfH);


            if(left < 0)             { left = 0; }
            if(right > imageWidth-1) { right = imageWidth-1; }
            if(top < 0)              { top = 0; }
            if(bot > imageHeight-1)  { bot = imageHeight-1; }

            printf("%s: %.2f - (%0.2f,%0.2f)->(%0.2f,%0.2f)\n", names[idx_class], prob , left,top,right,bot);

            drawRectangleRGB(pixels,imageWidth,imageHeight, (unsigned char) red,(unsigned char) green, (unsigned char) blue, 3, left,top,right,bot);
            /*

			cv::Size szTxt = cv::getTextSize(names[idx_class], cv::FONT_HERSHEY_SIMPLEX, TEXT_SCALE, 1, NULL);
			cv::rectangle(im, cv::Rect(left, top-1-szTxt.height, szTxt.width, szTxt.height), cv::Scalar(red, green, blue), -1);
			//cv::putText(im, names[idx_class], cv::Point(left, top-2), cv::FONT_HERSHEY_SIMPLEX, TEXT_SCALE, cv::Scalar(255, 255, 255), 2);
			cv::putText(im, names[idx_class], cv::Point(left, top-2), cv::FONT_HERSHEY_SIMPLEX, TEXT_SCALE, cv::Scalar(0, 0, 0), 2);
			cv::rectangle(im, cv::Rect(left, top, right-left, bot-top), cv::Scalar(red, green, blue), 2);*/
        }
    }
}




//https://github.com/TLESORT/YOLO-TensorRT-GIE-/blob/master/YOLODraw.cpp
void do_nms_sort(struct box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    struct sortable_bbox *s = (struct sortable_bbox*)calloc(total, sizeof(struct sortable_bbox));

    for(i = 0; i < total; ++i)
      {
        s[i].index = i;
        s[i].idx_class = 0;
        s[i].probs = probs;
      }

    for(k = 0; k < classes; ++k)
     {
        for(i = 0; i < total; ++i)
        {
            s[i].idx_class = k;
        }

        qsort(s, total, sizeof(struct sortable_bbox), nms_comparator);

        for(i = 0; i < total; ++i)
        {
            if(probs[s[i].index][k] == 0) continue;
            struct box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                struct box b = boxes[s[j].index];
                if (box_iou(&a, &b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }


    free(s);
}



//https://github.com/Guanghan/darknet/blob/master/src/yolo.c#L95
void  convert_yolo_detections(float *predictions, int classes, int num, int side, float w, float h, float thresh, float **probs,struct  box *boxes, int only_objectness)
{
// Interpret the output from a single inference of TinyYolo (GetResult)
// and filter out objects/boxes with low probabilities.
// output is the array of floats returned from the API GetResult but converted
// to float32 format.
// input_image_width is the width of the input image
// input_image_height is the height of the input image
// Returns a list of lists. each of the inner lists represent one found object and contain
// the following 6 values:
//    string that is network classification ie 'cat', or 'chair' etc
//    float value for box center X pixel location within source image
//    float value for box center Y pixel location within source image
//    float value for box width in pixels within source image
//    float value for box height in pixels within source image
//    float value that is the probability for the network classification.

     //classes = 20
     //num = 2
     //side = 7
    int i,j,n;
    for (i = 0; i < side*side; ++i)
      {
        int row = i / side;
        int col = i % side;

        for(n = 0; n < num; ++n)
        {
            int index = i*num + n;
            int p_index = side*side*classes /*980*/ + i*num + n;
            float scale = predictions[p_index];

            //printf("side : %d , index : %d , p_index : %d , prediction : %f\n",side,index,p_index,scale);
            int box_index = (side*side*(classes + num)) + (i*num + n)*4;
            boxes[index].x = (float) (predictions[box_index + 0] + col) / (side * w);
            boxes[index].y = (float) (predictions[box_index + 1] + row) / (side * h);
            boxes[index].w = (predictions[box_index + 2]*predictions[box_index + 2]) * w;
            boxes[index].h = (predictions[box_index + 3]*predictions[box_index + 3]) * h;

			for(j = 0; j < classes; ++j)
               {
				int class_index = i*classes;
				float prob = scale*predictions[class_index+j]; //


                if (prob > thresh) { probs[index][j] = prob; } else
                                   { probs[index][j] = 0.0;  }
			   }

			if(only_objectness)
               {
				probs[index][0] = scale;
			   }
        }
    }
}



int processTinyYOLO(struct labelContents * labels, float * results , unsigned int resultsLength ,char * pixels, unsigned int imageWidth, unsigned int imageHeight , float minimumConfidence)
{
    printf("got back %u resultData \n", resultsLength );
    struct box *boxes = (struct box*)calloc(NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES, sizeof(struct box));
	float **probs = (float**)calloc(NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES, sizeof(float *));
	for(int j = 0; j < NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES; ++j)
		probs[j] = (float*)calloc(NUM_CLASSES, sizeof(float *));

	convert_yolo_detections(
                             results,          //predictions
                             NUM_CLASSES,      //classes
                             NUM_TOP_CLASSES,  //num
                             NUM_CELLS,        //side
                             imageWidth,                //w
                             imageHeight,                //h
                             minimumConfidence,//threshold
                             probs,            //probability Output
                             boxes,            //Objects Output
                             0                 //onlyObjectness
                            );


	do_nms_sort(
                 boxes, //boxes
                 probs, //probabilities
                 NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES, //total
                 NUM_CLASSES,  //classes
                 minimumConfidence   //threshold
                );


    draw_detections( pixels,imageWidth, imageHeight ,
                     NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES, minimumConfidence, boxes, probs,(const char **) labels->content);


    free(boxes);
      for(int j = 0; j < NUM_CELLS*NUM_CELLS*NUM_TOP_CLASSES; ++j)
      free (probs[j]);
    free(probs);

 return 1;
/*
 printf("got back %u resultData \n", resultsLength );
 int i=0;
 float x,y,w,h,confidence,category;

 for (i=0; i<resultsLength/6; i++)
   {
    #define TRANSPOSED 0

    #if TRANSPOSED
     category   =  results[(i+0)*6];
     x =  results[(i+1)*6];
     y =  results[(i+2)*6];
     w =  results[(i+3)*6]/2;
     h =  results[(i+4)*6]/2;
     confidence =  results[(i+5)*6];
    #else
     category   =  results[(i*6)+0];
     x =  results[(i*6)+1];
     y =  results[(i*6)+2];
     w =  results[(i*6)+3]/2;
     h =  results[(i*6)+4]/2;
     confidence =  results[(i*6)+5];
    #endif

	float xmin = x-w , xmax = x+w , ymin = y-h , ymax = y+h;
    if (xmin<0)          { xmin = 0; }
    if (ymin<0)          { ymin = 0; }
    if (xmax>imageWidth) { xmax = imageWidth;  }
    if (ymax>imageHeight){ ymax = imageHeight; }


    //x*=imageWidth; y*=imageHeight;
    //w*=imageWidth; h*=imageHeight;

    //xmin*=imageWidth; ymin*=imageHeight;
    //xmax*=imageWidth; ymax*=imageHeight;

    if (confidence>minimumConfidence)
     {
      printf("%u> ",i);
      printf("cat=%0.2f ",category);
      printf("x1=%0.2f ",xmin);
      printf("y1=%0.2f ",ymin);
      printf("x2=%0.2f ",xmax);
      printf("y2=%0.2f ",ymax);
      printf("conf=%0.2f\n",confidence);
     }
   }

   */
 return 1;
}
