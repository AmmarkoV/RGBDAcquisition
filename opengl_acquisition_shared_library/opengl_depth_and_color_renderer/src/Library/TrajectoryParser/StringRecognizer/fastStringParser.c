#include "fastStringParser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

struct fastStringParser * fspHTTPHeader = 0;

#define MAXIMUM_FILENAME_WITH_EXTENSION 1024
#define MAXIMUM_LINE_LENGTH 1024
#define MAXIMUM_LEVELS 123
#define ACTIVATED_LEVELS 3

char acceptedChars[]="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_";



void convertTo_ENUM_ID(char *sPtr)
{
  unsigned int source=0 , target=0 , holdIt=0;

  while ( (sPtr[source] != 0 ) && (sPtr[target] != 0 ) )
   {
     sPtr[target] = toupper((unsigned char) sPtr[source]);

     if  (sPtr[source]=='_')  {  } else
     if  (sPtr[source]=='-')  { sPtr[target]='_'; } else
     if  (sPtr[source]=='.')  { sPtr[target]='_'; } else
     if  (
              ( (sPtr[source]>='A') && (sPtr[source]<='Z' ) )  ||
              ( (sPtr[source]>='0') && (sPtr[source]<='9' ) )
         )
     {
       //Will just copy those chars as they are
     } else
     {
       holdIt=1;
      ++target;
      sPtr[source]=sPtr[target];
      ++target;
     }

    if (!holdIt) { ++source; ++target; }
     holdIt=0;
   }
}



int fastStringParser_addString(struct fastStringParser * fsp, char * str)
{
  //TODO : Check here if there are smaller strings with the same prefix as us declared before
  //here if found then swap them with current word


  unsigned int ourNum = fsp->stringsLoaded++;
  fsp->contents[ourNum].strLength=strlen(str);

  if ( (ourNum==0) || (fsp->shortestStringLength > fsp->contents[ourNum].strLength) )
      { fsp->shortestStringLength = fsp->contents[ourNum].strLength; }

  if ( (ourNum==0) || (fsp->longestStringLength < fsp->contents[ourNum].strLength) )
      { fsp->longestStringLength  = fsp->contents[ourNum].strLength; }

  fsp->contents[ourNum].str = (char *) malloc(sizeof(char) * (fsp->contents[ourNum].strLength+1) );
  if (fsp->contents[ourNum].str != 0 )
  {
    strncpy(fsp->contents[ourNum].str,str,fsp->contents[ourNum].strLength);
    fsp->contents[ourNum].str[fsp->contents[ourNum].strLength]=0; // Null terminator
  } else
  {
    return 0;
  }


  fsp->contents[ourNum].strIDFriendly = (char *) malloc(sizeof(char) * (fsp->contents[ourNum].strLength+1) );
  if (fsp->contents[ourNum].strIDFriendly != 0 )
  {
    strncpy(fsp->contents[ourNum].strIDFriendly,str,fsp->contents[ourNum].strLength);
    fsp->contents[ourNum].strIDFriendly[fsp->contents[ourNum].strLength]=0; // Null terminator
    convertTo_ENUM_ID(fsp->contents[ourNum].strIDFriendly);
  } else
  {
    if (fsp->contents[ourNum].str!=0) { free(fsp->contents[ourNum].str); fsp->contents[ourNum].str=0; }
    return 0;
  }

  return 0;
}

struct fastStringParser *  fastStringParser_initialize(unsigned int totalStrings)
{
   fspHTTPHeader = (struct fastStringParser * ) malloc(sizeof( struct fastStringParser ));
   if (fspHTTPHeader== 0 ) { return 0; }

   fspHTTPHeader->stringsLoaded = 0;
   fspHTTPHeader->MAXstringsLoaded = totalStrings;
   fspHTTPHeader->contents = (struct fspString * ) malloc(sizeof( struct fspString )*fspHTTPHeader->MAXstringsLoaded);
   fspHTTPHeader->shortestStringLength = 0;
   fspHTTPHeader->longestStringLength = 0;

   if (fspHTTPHeader->contents== 0 ) { return 0; }

  return fspHTTPHeader;
}


int fastStringParser_hasStringsWithNConsecutiveChars(struct fastStringParser * fsp,unsigned int * resStringResultIndex, char * Sequence,unsigned int seqLength)
{
  int res = 0;
  unsigned int i=0,count = 0,correct=0;
  for (i=0; i<fsp->stringsLoaded; i++)
  {
    char * str1 = fsp->contents[i].str;
    char * str2 = Sequence;
    correct=0;
    for ( count=0; count<seqLength; count++ )
    {
      if (*str1==*str2) { ++correct; }
      ++str1;
      ++str2;
    }


    if ( correct == seqLength ) {
                                   *resStringResultIndex = i;
                                   ++res;
                                   //fprintf(stderr,"Comparing %s with %s : ",Sequence,fsp->contents[i].str);
                                   //fprintf(stderr,"HIT\n");
                                } else
                                { /*fprintf(stderr,"MISS\n");*/ }

  }
  return res ;
}


unsigned int fastStringParser_countStringsForNextChar(struct fastStringParser * fsp,unsigned int * resStringResultIndex,char * Sequence,unsigned int seqLength)
{
 unsigned int res=0;
 Sequence[seqLength+1]=0;
 Sequence[seqLength]='A';

 unsigned int curCh=0;
 while (curCh <= strlen(acceptedChars))
 {
   Sequence[seqLength] =  acceptedChars[curCh];
   res+=fastStringParser_hasStringsWithNConsecutiveChars(fsp,resStringResultIndex,Sequence,seqLength+1);
   ++curCh;
  }

  Sequence[seqLength]=0;
  //fprintf(stderr,"%u strings with prefix %s ( length %u ) \n",res,Sequence,seqLength);

  return res;
}

void addLevelSpaces(FILE * fp , unsigned int level)
{
  int i=0;
  for (i=0; i<level*4; i++)
  {
    fprintf(fp," ");
  }
}



int printIfAllPossibleStrings(FILE * fp , struct fastStringParser * fsp , char * Sequence,unsigned int seqLength)
{
  unsigned int i=0,count=0 , correct = 0 , results =0 ;
  for (i=0; i<fsp->stringsLoaded; i++)
  {
    char * str1 = fsp->contents[i].str;
    char * str2 = Sequence;
    correct=0;
    for ( count=0; count<seqLength; count++ )
    {
      if (*str1==*str2) { ++correct; }
      ++str1;
      ++str2;
    }

    if ( correct == seqLength ) {
                                  addLevelSpaces(fp , seqLength);
                                  if (results>0) { fprintf(fp," else "); }
                                  fprintf(fp," if ( (strLength >= %u )&& ( strncasecmp(str,\"%s\",%u) == 0 ) ) { return %s_%s; } \n",
                                          (unsigned int) strlen(fsp->contents[i].str),
                                          fsp->contents[i].str ,
                                          (unsigned int) strlen(fsp->contents[i].str) ,
                                          fsp->functionName,
                                          fsp->contents[i].strIDFriendly );

                                  ++results;
                                }
  }
  return 1;
}



int printAllEnumeratorItems(FILE * fp , struct fastStringParser * fsp,char * functionName)
{
  fprintf(fp,"enum { \n");
  fprintf(fp," %s_EMPTY=0,\n",fsp->functionName);

  unsigned int i=0;
  for (i=0; i<fsp->stringsLoaded; i++)
  {
    fprintf(fp," %s_%s, // %u \n",fsp->functionName,fsp->contents[i].strIDFriendly,i+1);
  }

  fprintf(fp," %s_END_OF_ITEMS\n",fsp->functionName);

  fprintf(fp,"};\n\n");

  return 1;
}



int recursiveTraverser(FILE * fp,struct fastStringParser * fsp,char * functionName,char * cArray,unsigned int level)
{
  if (level>=ACTIVATED_LEVELS) { return 0; }

  unsigned int resStringResultIndex=0;
  unsigned int nextLevelStrings=fastStringParser_countStringsForNextChar(fsp,&resStringResultIndex,cArray,level);

  if ( nextLevelStrings>1 )
     {
      unsigned int cases=0;
      addLevelSpaces(fp , level);
      fprintf(fp," switch (toupper(str[%u])) { \n",level);

      cArray[level]='A';
      cArray[level+1]=0;
      //TODO: Add '-' character for strings like IF-MODIFIED-ETC
      unsigned int curCh=0;
      while (curCh <= strlen(acceptedChars))
       {
        cArray[level]=acceptedChars[curCh];
        if ( fastStringParser_hasStringsWithNConsecutiveChars(fsp,&resStringResultIndex,cArray,level+1)  )
        {
          addLevelSpaces(fp , level);

          if (cArray[level]==0)
              { fprintf(fp," case 0 : /*Null Terminator*/ \n"); } else
              { fprintf(fp," case \'%c\' : \n",cArray[level]);  }
           if ( level < ACTIVATED_LEVELS-1 ) { recursiveTraverser(fp,fsp,functionName,cArray,level+1); } else
                                             { printIfAllPossibleStrings(fp , fsp , cArray, level+1); }
          addLevelSpaces(fp , level);
          fprintf(fp," break; \n");
          ++cases;
        }

         ++curCh;
       }
       cArray[level]=0;


       if (cases==0) { fprintf(fp,"//BUG :  nextLevelStrings were supposed to be non-zero"); }

       addLevelSpaces(fp , level);
       fprintf(fp,"}; \n");
     } else
     if ( nextLevelStrings==1 )
     {
       addLevelSpaces(fp , level);
       fprintf(fp," if (strLength<%u) { return 0; } \n", (unsigned int ) strlen(fsp->contents[resStringResultIndex].str));
       addLevelSpaces(fp , level);
       fprintf(fp," if ( strncasecmp(str,\"%s\",%u) == 0 ) { return %s_%s; } \n",
                   fsp->contents[resStringResultIndex].str ,
                   (unsigned int) strlen(fsp->contents[resStringResultIndex].str),
                   fsp->functionName,
                   fsp->contents[resStringResultIndex].strIDFriendly );
     }
     else
     {
       fprintf(fp," //Error ( %s ) \n",fsp->contents[resStringResultIndex].str);
     }

 return 1;
}




int export_C_Scanner(struct fastStringParser * fsp,char * functionName)
{
  if (fsp==0) { fprintf(stderr,"export_C_Scanner called with empty string parser\n"); return 0; }
  if (functionName==0) { fprintf(stderr,"export_C_Scanner called with empty function name\n"); return 0; }

  unsigned int functionNameLength = strlen(functionName);
  fsp->functionName  = (char* ) malloc(sizeof(char) * (1+functionNameLength));
  if (fsp->functionName==0) { fprintf(stderr,"Could not allocate memory for function name\n"); return 0; }
  strncpy(fsp->functionName,functionName,functionNameLength);
  fsp->functionName[functionNameLength]=0;

  convertTo_ENUM_ID(fsp->functionName);


  char filenameWithExtension[MAXIMUM_FILENAME_WITH_EXTENSION+1]={0};


  //PRINT OUT THE HEADER
  snprintf(filenameWithExtension,MAXIMUM_FILENAME_WITH_EXTENSION,"%s.h",functionName);
  FILE * fp = fopen(filenameWithExtension,"wb");
  if (fp == 0) { fprintf(stderr,"Could not open input file %s\n",functionName); return 0; }
  fflush(fp);
  fprintf(fp,"\n");


  fprintf(fp,"/** @file %s.h\n",functionName);
  fprintf(fp,"* @brief A tool that scans for a string in a very fast and robust way\n");
  fprintf(fp,"* @author Ammar Qammaz (AmmarkoV)\n");
  fprintf(fp,"*/\n\n");

  fprintf(fp,"#ifndef %s_H_INCLUDED\n",fsp->functionName);
  fprintf(fp,"#define %s_H_INCLUDED\n\n\n",fsp->functionName);


      fprintf(fp,"/** @brief Enumerator for the IDs of %s so we can know what the result was*/\n",functionName);
      printAllEnumeratorItems(fp, fsp, functionName);


  fprintf(fp,"\n\n/** @brief Scan a string for one of the words of the %s word set\n",functionName);
  fprintf(fp,"* @ingroup stringParsing\n");
  fprintf(fp,"* @param Input String , to be scanned\n");
  fprintf(fp,"* @param Length of Input String\n");
  fprintf(fp,"* @retval See above enumerator*/\n");
  fprintf(fp," int scanFor_%s(const char * str,unsigned int strLength); \n\n",functionName);
  fprintf(fp,"#endif\n");
  fclose(fp);


  //PRINT OUT THE MAIN FILE

  snprintf(filenameWithExtension,MAXIMUM_FILENAME_WITH_EXTENSION,"%s.c",functionName);
  fp = fopen(filenameWithExtension,"wb");
  if (fp == 0) { fprintf(stderr,"Could not open input file %s\n",functionName); return 0; }

  fflush(fp);
  fprintf(fp,"\n");

  char cArray[MAXIMUM_LEVELS]={0};
  int i=0;
  for (i=0; i<MAXIMUM_LEVELS; i++ ) { cArray[i]=0;/*'A';*/ }


  time_t t = time(NULL);
  struct tm tm = *localtime(&t);


  fprintf(fp,"/* \
                 \nThis file was automatically generated @ %02d-%02d-%02d %02d:%02d:%02d using StringRecognizer \
                 \nhttps://github.com/AmmarkoV/AmmarServer/tree/master/src/StringRecognizer\
                 \nPlease note that changes you make here may be automatically overwritten \
                 \nif the String Recognizer generator runs again..!\
              \n */ \n\n" ,
          tm.tm_mday, tm.tm_mon + 1, tm.tm_year + 1900,   tm.tm_hour, tm.tm_min, tm.tm_sec);


  fprintf(fp,"#include <stdio.h>\n");
  fprintf(fp,"#include <string.h>\n");
  fprintf(fp,"#include <ctype.h>\n");
  fprintf(fp,"#include \"%s.h\"\n\n",functionName);

  fprintf(fp,"int scanFor_%s(const char * str,unsigned int strLength) \n{\n",functionName);

     fprintf(fp," if (str==0) { return 0; } \n");
     fprintf(fp," if (strLength<%u) { return 0; } \n\n",fsp->shortestStringLength);
     recursiveTraverser(fp,fsp,functionName,cArray,0);

  fprintf(fp," return 0;\n");
  fprintf(fp,"}\n");

  /*
  fprintf(fp,"\n\nint main(int argc, char *argv[]) \n {\n");
  fprintf(fp,"  if (argc<1) { fprintf(stderr,\"No parameter\\n\"); return 1; }\n");
  fprintf(fp,"  if ( scanFor_%s(argv[0]) ) { fprintf(stderr,\"Found it\"); } \n  return 0; \n }\n",functionName);
*/

  fclose(fp);

  return 1;
}






struct fastStringParser * fastSTringParser_createRulesFromFile(char* filename,unsigned int totalStrings)
{
  FILE * fp = fopen(filename,"r");
  if (fp == 0) { fprintf(stderr,"Could not open input file %s\n",filename); return 0; }

  struct fastStringParser *  fsp  = fastStringParser_initialize(totalStrings);
  if (fsp==0) { fclose(fp); return 0; }

  char line[MAXIMUM_LINE_LENGTH]={0};
  unsigned int lineLength=0;
  while (fgets(line,MAXIMUM_LINE_LENGTH,fp)!=0)
  {
      lineLength = strlen(line);
      if ( lineLength > 0 )
        {
         if (line[lineLength-1]==10) { line[lineLength-1]=0; --lineLength; } else
         if (line[lineLength-1]==13) { line[lineLength-1]=0; --lineLength; }
        }
      if ( lineLength > 1 )
        {
         if (line[lineLength-2]==10) { line[lineLength-2]=0; --lineLength; } else
         if (line[lineLength-2]==13) { line[lineLength-2]=0; --lineLength; }
        }

    //fprintf(stderr,"LINE : `%s`\n",line);
    fastStringParser_addString(fsp,line);
  }
  fclose(fp);


  return fsp;
}





int fastStringParser_close(struct fastStringParser * fsp)
{

    fprintf(stderr,"TODO: Deallocate here\nClosing Fast String Parser\n");
    return 1;
}




