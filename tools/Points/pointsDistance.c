/** @file pointsDistance.c
*   @brief  A small tool that calculates the distance between 2 points
*   @author Ammar Qammaz (AmmarkoV)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <locale.h>


int forceUSLocaleToKeepOurSanity()
{
   setlocale(LC_ALL, "en_US.UTF-8");
   setlocale(LC_NUMERIC, "en_US.UTF-8");
   return 1;
}

int main(int argc, char *argv[])
{
    forceUSLocaleToKeepOurSanity();
    if (argc<6) { printf("%s X Y Z , X Y Z , you did not provide 6 arguments\n",argv[0]); return 1; }

    double xA = atof(argv[1]);
    double yA = atof(argv[2]);
    double zA = atof(argv[3]);

    double xB = atof(argv[4]);
    double yB = atof(argv[5]);
    double zB = atof(argv[6]);

    double distance = sqrt( ((xA-xB)*(xA-xB)) + ((yA-yB)*(yA-yB)) + ((zA-zB)*(zA-zB)) );

    printf("%f\n",distance);

    return 0;
}
