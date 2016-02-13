
/** @file locationService.h
 *  @brief A wrapper for libgpsd that provides location service for grabbing.
 *
 *
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef LOCATIONSERVICE_H_INCLUDED
#define LOCATIONSERVICE_H_INCLUDED


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Start Location Services , this assumes a running gpsd instance on the background
 * @ingroup locationService
 * @retval 1=Success,0=Failure
 */
int startLocationServices();

/**
 * @brief  Stop Location Services
 * @ingroup locationService
 * @retval 1=Success,0=Failure
 */
int stopLocationServices();

/**
 * @brief  Talk to libgpsd ( and connected gps devices ) and get a new fix
 * @ingroup locationService
 * @retval 1=Success,0=Failure
 */
int pollLocationServices();


/**
 * @brief  Check if location service is active and running
 * @ingroup locationService
 * @retval 1=Success,0=Failure
 */
int locationServicesOK();


/**
 * @brief  Number of sattelites currently used
 * @ingroup locationService
 * @retval 1=Success,0=Failure
 */
unsigned int sattelitesUsed();



/**
 * @brief  Get Current Altitude
 * @ingroup locationService
 * @retval Altitude,0.0=Failure
 */
double getAlt();
/**
 * @brief  Get Current Latitude
 * @ingroup locationService
 * @retval Latitude,0.0=Failure
 */
double getLat();
/**
 * @brief  Get Current Longitude
 * @ingroup locationService
 * @retval Longitude,0.0=Failure
 */
double getLon();

double getSpeed();
double getClimb();
double getBearing();


/**
 * @brief  Check if fix is 3D
 * @ingroup locationService
 * @retval 1=3DFix,0=NotA3DFix
 */
int fixTypeIs3D();
/**
 * @brief  Check if fix is 2D
 * @ingroup locationService
 * @retval 1=2DFix,0=NotA2DFix
 */
int fixTypeIs2D();
/**
 * @brief  Check if we don't have a fix
 * @ingroup locationService
 * @retval 1=WeHaveNoFix,0=WeHaveAFix
 */
int nofix();

#ifdef __cplusplus
}
#endif

#endif // LOCATIONSERVICE_H_INCLUDED
