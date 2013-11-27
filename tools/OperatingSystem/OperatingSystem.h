#ifndef OPERATINGSYSTEM_H_INCLUDED
#define OPERATINGSYSTEM_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif

int copyDirectoryListItem(int itemNum , char * directoryList , char * output, unsigned int maxOutput);
int listDirectory(char * directory , char * output, unsigned int maxOutput);




#ifdef __cplusplus
}
#endif


#endif // OPERATINGSYSTEMG_H_INCLUDED
