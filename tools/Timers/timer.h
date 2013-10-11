#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED



enum VisCortTimerList
{
   UNUSED_TIMER_0 = 0 ,
   UNUSED_TIMER_1 ,
   UNUSED_TIMER_2 ,
   UNUSED_TIMER_3 ,
   UNUSED_TIMER_4 ,
   UNUSED_TIMER_5 ,
   //<- 0-5 reserved for use by clients :P
   FRAME_SNAP_DELAY  ,
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

unsigned long GetTickCountInternal();

#endif // VISCORTEXTIMER_H_INCLUDED
