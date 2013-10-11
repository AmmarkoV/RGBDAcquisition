#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED



enum VisCortTimerList
{
   FRAME_SNAP_DELAY = 0 ,
   FRAME_PASS_TO_TARGET_DELAY ,
   TOTAL_TIMERS
};


void StartTimer( unsigned int timer_num );
unsigned int EndTimer( unsigned int timer_num );
unsigned int GetLastTimer( unsigned int timer_num );
unsigned int GetAverageTimer( unsigned int timer_num );
unsigned int GetTimesTimerTimed( unsigned int timer_num );
float GetFPSTimer( unsigned int timer_num );

void VisCortxMillisecondsSleep(unsigned int milliseconds);
void VisCortxMicrosecondsSleep(unsigned int microseconds);
#endif // VISCORTEXTIMER_H_INCLUDED
