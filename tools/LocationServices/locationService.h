#ifndef LOCATIONSERVICE_H_INCLUDED
#define LOCATIONSERVICE_H_INCLUDED


#ifdef __cplusplus
extern "C" {
#endif

int startLocationServices();
int stopLocationServices();
int pollLocationServices();


unsigned int sattelitesUsed();

double getAlt();
double getLat();
double getLon();

double getSpeed();
double getClimb();
double getBearing();


int fixTypeIs3D();
int fixTypeIs2D();
int nofix();


#ifdef __cplusplus
}
#endif

#endif // LOCATIONSERVICE_H_INCLUDED
