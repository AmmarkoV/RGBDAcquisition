/*
 * This file is Copyright (c) 2010 by the GPSD project
 * BSD terms apply: see the file COPYING in the distribution root for details.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <syslog.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <libgen.h>
#include <signal.h>
#ifndef S_SPLINT_S
#include <unistd.h>
#endif /* S_SPLINT_S */

#include <gps.h>



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

#define SOCKET_EXPORT_ENABLE 1

struct fixsource_t
/* describe a data source */
{
 char *spec;     /* pointer to actual storage */
 char *server;
 char *port;
 char *device;
};

#define NITEMS(x) (int)(sizeof(x)/sizeof(x[0])) /* from gpsd.h-tail */

#ifdef S_SPLINT_S
extern struct tm *gmtime_r(const time_t *, /*@out@*/ struct tm *tp);
#endif /* S_SPLINT_S */

static char *progname;
static struct fixsource_t source;

/**************************************************************************
 *
 * Transport-layer-independent functions
 *
 **************************************************************************/

int working=0;
static struct gps_data_t gpsdata;
static FILE *logfile;
static bool intrack = false;
static time_t timeout = 5;	/* seconds */
static double minmove = 0;	/* meters */

static void quit_handler(int signum)
{
    /* don't clutter the logs on Ctrl-C */
    if (signum != SIGINT)
	syslog(LOG_INFO, "exiting, signal %d received", signum);
    print_gpx_footer();
    (void)gps_close(&gpsdata);
    exit(0);
}

#ifdef SOCKET_EXPORT_ENABLE
/**************************************************************************
 *
 * Doing it with sockets
 *
 **************************************************************************/

/*@-mustfreefresh -compdestroy@*/
static int socket_mainloop(void)
{
    unsigned int flags = WATCH_ENABLE;

    if (gps_open(source.server, source.port, &gpsdata) != 0) {
	(void)fprintf(stderr,
		      "%s: no gpsd running or network error: %d, %s\n",
		      progname, errno, gps_errstr(errno));
	exit(1);
    }

    if (source.device != NULL)
	flags |= WATCH_DEVICE;
    (void)gps_stream(&gpsdata, flags, source.device);

    print_gpx_header();
    for (;;) {
	if (!gps_waiting(&gpsdata, 5000000)) {
	    (void)fprintf(stderr, "%s: error while waiting\n", progname);
	    break;
	} else {
	    (void)gps_read(&gpsdata);
	    conditionally_log_fix(&gpsdata);
	}
    }
    (void)gps_close(&gpsdata);
    return 0;
}
/*@+mustfreefresh +compdestroy@*/
#endif /* SOCKET_EXPORT_ENABLE */


/**************************************************************************
 *
 * Main sequence
 *
 **************************************************************************/



int pollLocationServices()
{
 if (!working) { return 0; }

  if (!gps_waiting(&gpsdata, 5000000))
     {
	    fprintf(stderr, "LoacationService : error while waiting\n");
        return 0;
     } else
	 {
	    fprintf(stderr,"!");
	    (void)gps_read(&gpsdata);
	    // This is not needed => //conditionally_log_fix(&gpsdata);
        return 1;
	 }

 return 0;
}

int startLocationServices()
{
  if (working) { return 0.0; }

    gpsdata.status = STATUS_NO_FIX;
    gpsdata.satellites_used = 0;
    gps_clear_fix(&(gpsdata.fix));
    gps_clear_dop(&(gpsdata.dop));

    gpsdata.fix.latitude=0;
    gpsdata.fix.longitude=0;

    logfile=stdout;
	gpsd_source_spec(NULL, &source);


    unsigned int flags = WATCH_ENABLE;

    if (gps_open("127.0.0.1" , "2947", &gpsdata) != 0)
        { fprintf(stderr,"LoacationService : no gpsd running or network error: %d, %s\n", errno, gps_errstr(errno)); return 0; }

    if (source.device != NULL)
	flags |= WATCH_DEVICE;
    gps_stream(&gpsdata, flags, source.device);


    fprintf(stderr,MAGENTA "Polling location services to determine if they will be used in this run : \n" NORMAL);
    fprintf(stderr,MAGENTA "If you don't want to wait use `sudo service gpsd stop`\n" NORMAL);
    unsigned int tries=3;
    unsigned int success=0;
    unsigned int i=0;
    working=1; //lets assume that we are working so that  pollLocationServices will work
    for (i=0; i<tries; i++)
    {
        success+=pollLocationServices();
        usleep(1000*1000);
    }

    if (success==0)
    {
     fprintf(stderr,RED "Location services failed to provide gps coordinates beeing switched off\n" NORMAL);
     working=0;
    } else
    if (success<tries)
    {
     fprintf(stderr,YELLOW "Location services failed to provide gps coordinates beeing switched off\n" NORMAL);
     working=1;
    } else
    if (success==tries)
    {
     fprintf(stderr,GREEN "Location services working fine.. :) \n" NORMAL);
     working=1;
    }

 return working;
}



int stopLocationServices()
{
 if (!working) { return 0; }
 gps_close(&gpsdata);
 working=0;
return 1;
}

int locationServicesOK()
{
  return working;
}

unsigned int sattelitesUsed()
{
 if (!working) { return 0; }
 return gpsdata.satellites_used;
}


double getAlt()
{
 if (!working) { return 0.0; }
 return gpsdata.fix.altitude;
}

double getLat()
{
 if (!working) { return 0.0; }
 return gpsdata.fix.latitude;
}

double getLon()
{
 if (!working) { return 0.0; }
 return gpsdata.fix.longitude;
}



double getSpeed()
{
 if (!working) { return 0.0; }
 return gpsdata.fix.speed;
}


double getClimb()
{
 if (!working) { return 0.0; }
 return gpsdata.fix.climb;
}

double getBearing()
{
 if (!working) { return 0.0; }
 return gpsdata.fix.track;
}

int fixTypeIs3D()
{
 if (!working) { return 0; }
 return  (gpsdata.fix.mode==MODE_3D);
}

int fixTypeIs2D()
{
 if (!working) { return 0; }
 return  (gpsdata.fix.mode==MODE_2D);
}

int nofix()
{
 if (!working) { return 1; }
 return  (gpsdata.fix.mode==MODE_NO_FIX);
}

