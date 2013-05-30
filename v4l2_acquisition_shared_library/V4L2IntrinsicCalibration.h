#ifndef V4L2INTRINSICCALIBRATION_H_INCLUDED
#define V4L2INTRINSICCALIBRATION_H_INCLUDED

unsigned int PrecalcResectioning(unsigned int * frame ,     unsigned int width , unsigned int height ,
                                                            double fx,double fy , double cx,double cy ,
                                                            double k1,double k2 , double p1,double p2 , double k3   );

#endif // V4L2INTRINSICCALIBRATION_H_INCLUDED
