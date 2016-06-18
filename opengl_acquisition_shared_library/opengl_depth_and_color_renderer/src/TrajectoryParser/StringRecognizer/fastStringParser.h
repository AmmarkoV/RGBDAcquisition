/** @file fastStringParser.h
* @brief A tool that converts a file with words ( each word on a new line ) to C code ( see automata ) for fast string checking
* @bug In case the declarations have shared prefixes and the shortest prefix is stated first they will also get recognized first so be careful
*
* @author Ammar Qammaz (AmmarkoV)
*/

#ifndef FASTSTRINGPARSER_H_INCLUDED
#define FASTSTRINGPARSER_H_INCLUDED

/** @brief Internal Structure to hold a string and its id for further processing */
struct fspString
{
  char * str;
  char * strIDFriendly;
  unsigned int strLength;

};

/** @brief Internal Structure that holds all the string parser context */
struct fastStringParser
{
  struct fspString * contents;
  unsigned int stringsLoaded;
  unsigned int MAXstringsLoaded;

  char * functionName;

  unsigned int shortestStringLength;
  unsigned int longestStringLength;
};

/**
* @brief Export a C Scanner source code
* @ingroup stringParsing
* @param Structure to hold all the intermediate state
* @param Name of the current function
* @retval 1=Success,0=Failure
*/
int export_C_Scanner(struct fastStringParser * fsp,char * filename);


/**
* @brief Read a file and create C files that parse the input
* @ingroup stringParsing
* @param Filename of the current function
* @param Total Number of Strings
* @retval fastStringParser context,0=Failure
*/
struct fastStringParser * fastSTringParser_createRulesFromFile(char* filename,unsigned int totalStrings);


/**
* @brief Destroy fast string parser
* @ingroup stringParsing
* @param Structure that holds the parser
* @retval 1=Success,0=Failure
*/
int fastStringParser_close(struct fastStringParser * fsp);

#endif // FASTSTRINGPARSER_H_INCLUDED
