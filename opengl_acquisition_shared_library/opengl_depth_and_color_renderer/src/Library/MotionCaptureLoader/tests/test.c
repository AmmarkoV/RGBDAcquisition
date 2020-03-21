#include "test.h"

#include <stdio.h>

int bvh_testConstrainRotations()
{
  /*
  fprintf(stderr,"Testing bvh_rotation constraint\n");
  unsigned int i=0;
  double angle = -720;
  for (i=0; i<1440; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Centered at 0:%0.2f | Flipped at 0:%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_constrainAngleCentered0(angle,0),
    bvh_constrainAngleCentered0(angle,1)
    //bvh_constrainAngleCentered180(angle)
    );
    angle=angle+1.0;
  }

  */


  fprintf(stderr,"Testing bvh_rotation front constraint\n");
  unsigned int i=0;
  double angle = -45;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Front Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_FRONT_ORIENTATION)
    );
    angle=angle+1.0;
  }


  fprintf(stderr,"Testing bvh_rotation back constraint\n");
  i=0;
  angle = 135;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Back Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_BACK_ORIENTATION)
    );
    angle=angle+1.0;
  }


  fprintf(stderr,"Testing bvh_rotation right constraint\n");
  i=0;
  angle = 45;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Right Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_RIGHT_ORIENTATION)
    );
    angle=angle+1.0;
  }


  fprintf(stderr,"Testing bvh_rotation left constraint\n");
  i=0;
  angle = -135;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Left Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_LEFT_ORIENTATION)
    );
    angle=angle+1.0;
  }


 return 0;
}
