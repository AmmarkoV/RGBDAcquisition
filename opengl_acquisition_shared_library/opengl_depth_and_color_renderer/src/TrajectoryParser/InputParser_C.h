#ifndef _INPUTPARSER_C_H_
#define _INPUTPARSER_C_H_

#ifdef __cplusplus
extern "C" {
#endif


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define DELIM_MAX_MAX 6
#define CONTAINERS_MAX 1
#define MAX_COMPLICITY 4

#define MAX_MEMORY 256
#define MAX_STRING 2048

#define USE_SCANF 0




struct tokens
{
  /* START MEANS THE FIRST CHARACTER TO READ..!
     LENGTH MEANS total characters to be read
     i.e.   a _ S _ A _ M _ P _ L _ E
            0   1   2   3   4   5   6
     this token starts from 0 and has 7 characters length*/
  unsigned int token_start;
  unsigned int length;
};

struct guard_byte
{
  unsigned int checksum;
};

struct InputParserC
{
    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    struct guard_byte guardbyte1;
    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */

    unsigned int str_length;
    unsigned char local_allocation;
    char * str; /* String to process */

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    struct guard_byte guardbyte2;
    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    unsigned short cur_container_count;
    unsigned short max_container_count;
    char *container_start;
    char *container_end;

    unsigned short cur_delimeter_count;
    unsigned short max_delimeter_count;
    char *delimeters;

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    struct guard_byte guardbyte3;
    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */

    /* RESULT */
    unsigned int tokens_max;
    unsigned int tokens_count;
    struct tokens* tokenlist;

    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
    struct guard_byte guardbyte4;
    /* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
};

char * InputParserC_Version();

int InputParser_ClearNonCharacters(char * inpt , unsigned int length);
int InputParser_TrimCharactersStart(char * inpt , unsigned int length,char what2trim);
int InputParser_TrimCharactersEnd(char * inpt , unsigned int length,char what2trim);
int InputParser_TrimCharacters(char * inpt , unsigned int length,char what2trim);

void InputParser_DefaultDelimeters(struct InputParserC * ipc);
void InputParser_SetDelimeter(struct InputParserC * ipc,int num,char tmp);
char InputParser_GetDelimeter(struct InputParserC * ipc,int num);

struct InputParserC * InputParser_Create(unsigned int max_string_count,unsigned int max_delimiter_count);
void InputParser_Destroy(struct InputParserC * ipc);
unsigned char InputParser_SelfCheck(struct InputParserC * ipc);

unsigned char CheckWordNumOk(struct InputParserC * ipc,unsigned int num);

char InputParser_GetWordChar(struct InputParserC * ipc,unsigned int num,unsigned int pos);

/* WordCompare and WordCompareNoCase return 1 when strings match , 0 otherwise..*/
unsigned char InputParser_WordCompareNoCase(struct InputParserC * ipc,unsigned int num,char * word,unsigned int wordsize);
unsigned char InputParser_WordCompareNoCaseAuto(struct InputParserC * ipc,unsigned int num,char * word);
unsigned char InputParser_WordCompare(struct InputParserC * ipc,unsigned int num,char * word,unsigned int wordsize);
unsigned char InputParser_WordCompareAuto(struct InputParserC * ipc,unsigned int num,char * word);

unsigned int InputParser_GetWord(struct InputParserC * ipc,unsigned int num,char * wheretostore,unsigned int storagesize);
unsigned int InputParser_GetUpcaseWord(struct InputParserC * ipc,unsigned int num,char * wheretostore,unsigned int storagesize);
unsigned int InputParser_GetLowercaseWord(struct InputParserC * ipc,unsigned int num,char * wheretostore,unsigned int storagesize);
signed int InputParser_GetWordInt(struct InputParserC * ipc,unsigned int num);
float InputParser_GetWordFloat(struct InputParserC * ipc,unsigned int num);
unsigned int InputParser_GetWordLength(struct InputParserC * ipc,unsigned int num);

int InputParser_SeperateWords(struct InputParserC * ipc,char * inpt,char keepcopy);
int InputParser_SeperateWordsCC(struct InputParserC * ipc,const char * inpt,char keepcopy);
int InputParser_SeperateWordsUC(struct InputParserC * ipc,unsigned char * inpt,char keepcopy);


#ifdef __cplusplus
}
#endif


#endif

