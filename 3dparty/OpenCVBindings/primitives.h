#ifndef PRIMITIVES_H_INCLUDED
#define PRIMITIVES_H_INCLUDED



struct Point2D
{
  double x , y;
};



struct Point2DCorrespondance
{
  struct Point2D  * listSource;
  struct Point2D  * listTarget;
  unsigned int listCurrent;
  unsigned int listMax;
};







#endif // PRIMITIVES_H_INCLUDED
