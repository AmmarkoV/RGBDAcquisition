#ifndef LOCATIONSERVICE_H_INCLUDED
#define LOCATIONSERVICE_H_INCLUDED


#ifdef __cplusplus
extern "C" {
#endif



int startLocationServices();
int stopLocationServices();
int pollLocationServices();
double getLat();
double getLon();


#ifdef __cplusplus
}
#endif

#endif // LOCATIONSERVICE_H_INCLUDED
