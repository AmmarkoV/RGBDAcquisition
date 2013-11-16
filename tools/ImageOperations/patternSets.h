#ifndef PATTERNSETS_H_INCLUDED
#define PATTERNSETS_H_INCLUDED



struct PatternItem
{
   unsigned int totalTiles;
   unsigned int  value;
   unsigned char use;
   unsigned int acceptScore;
   struct Image * tile[256];
   char name[512];
};

struct PatternSet
{
    unsigned int totalPatterns;
    struct PatternItem pattern[256];
};

int addToPatternSet(struct PatternSet * set , char * name , unsigned int value , unsigned int score);
int dumpPatternSet(struct PatternSet * pattSet ,char * stage);
int emptyPatternSet(struct PatternSet * pattSet);

int compareToPatternSet(struct PatternSet * pattSet ,
                        unsigned char * screen , unsigned int screenWidth ,unsigned int screenHeight ,
                        unsigned int sX,unsigned int sY , unsigned int width ,unsigned int height ,
                        unsigned int maximumAcceptedScore,
                        unsigned int * pick);


#endif // PATTERNSETS_H_INCLUDED
