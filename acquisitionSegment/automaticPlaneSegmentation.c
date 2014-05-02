#include <stdio.h>
#include "automaticPlaneSegmentation.h"

#define MEMPLACE1(x,y,width) ( y * ( width ) + x )
#define ResultNormals 42


enum pShorthand
{
  X=0,Y,Z
};


enum pointShorthand
{
  Point1=0,
  Point2,
  Point3
};


enum signShorthand
{
  LT=0,
  EQ,
  GT
};


enum hashesBruteShorthand
{
LT_LT_LT_LT_LT_LT=0,
EQ_LT_LT_LT_LT_LT=1,
GT_LT_LT_LT_LT_LT=2,
LT_EQ_LT_LT_LT_LT=3,
EQ_EQ_LT_LT_LT_LT=4,
GT_EQ_LT_LT_LT_LT=5,
LT_GT_LT_LT_LT_LT=6,
EQ_GT_LT_LT_LT_LT=7,
GT_GT_LT_LT_LT_LT=8,
LT_LT_EQ_LT_LT_LT=9,
EQ_LT_EQ_LT_LT_LT=10,
GT_LT_EQ_LT_LT_LT=11,
LT_EQ_EQ_LT_LT_LT=12,
EQ_EQ_EQ_LT_LT_LT=13,
GT_EQ_EQ_LT_LT_LT=14,
LT_GT_EQ_LT_LT_LT=15,
EQ_GT_EQ_LT_LT_LT=16,
GT_GT_EQ_LT_LT_LT=17,
LT_LT_GT_LT_LT_LT=18,
EQ_LT_GT_LT_LT_LT=19,
GT_LT_GT_LT_LT_LT=20,
LT_EQ_GT_LT_LT_LT=21,
EQ_EQ_GT_LT_LT_LT=22,
GT_EQ_GT_LT_LT_LT=23,
LT_GT_GT_LT_LT_LT=24,
EQ_GT_GT_LT_LT_LT=25,
GT_GT_GT_LT_LT_LT=26,
LT_LT_LT_EQ_LT_LT=27,
EQ_LT_LT_EQ_LT_LT=28,
GT_LT_LT_EQ_LT_LT=29,
LT_EQ_LT_EQ_LT_LT=30,
EQ_EQ_LT_EQ_LT_LT=31,
GT_EQ_LT_EQ_LT_LT=32,
LT_GT_LT_EQ_LT_LT=33,
EQ_GT_LT_EQ_LT_LT=34,
GT_GT_LT_EQ_LT_LT=35,
LT_LT_EQ_EQ_LT_LT=36,
EQ_LT_EQ_EQ_LT_LT=37,
GT_LT_EQ_EQ_LT_LT=38,
LT_EQ_EQ_EQ_LT_LT=39,
EQ_EQ_EQ_EQ_LT_LT=40,
GT_EQ_EQ_EQ_LT_LT=41,
LT_GT_EQ_EQ_LT_LT=42,
EQ_GT_EQ_EQ_LT_LT=43,
GT_GT_EQ_EQ_LT_LT=44,
LT_LT_GT_EQ_LT_LT=45,
EQ_LT_GT_EQ_LT_LT=46,
GT_LT_GT_EQ_LT_LT=47,
LT_EQ_GT_EQ_LT_LT=48,
EQ_EQ_GT_EQ_LT_LT=49,
GT_EQ_GT_EQ_LT_LT=50,
LT_GT_GT_EQ_LT_LT=51,
EQ_GT_GT_EQ_LT_LT=52,
GT_GT_GT_EQ_LT_LT=53,
LT_LT_LT_GT_LT_LT=54,
EQ_LT_LT_GT_LT_LT=55,
GT_LT_LT_GT_LT_LT=56,
LT_EQ_LT_GT_LT_LT=57,
EQ_EQ_LT_GT_LT_LT=58,
GT_EQ_LT_GT_LT_LT=59,
LT_GT_LT_GT_LT_LT=60,
EQ_GT_LT_GT_LT_LT=61,
GT_GT_LT_GT_LT_LT=62,
LT_LT_EQ_GT_LT_LT=63,
EQ_LT_EQ_GT_LT_LT=64,
GT_LT_EQ_GT_LT_LT=65,
LT_EQ_EQ_GT_LT_LT=66,
EQ_EQ_EQ_GT_LT_LT=67,
GT_EQ_EQ_GT_LT_LT=68,
LT_GT_EQ_GT_LT_LT=69,
EQ_GT_EQ_GT_LT_LT=70,
GT_GT_EQ_GT_LT_LT=71,
LT_LT_GT_GT_LT_LT=72,
EQ_LT_GT_GT_LT_LT=73,
GT_LT_GT_GT_LT_LT=74,
LT_EQ_GT_GT_LT_LT=75,
EQ_EQ_GT_GT_LT_LT=76,
GT_EQ_GT_GT_LT_LT=77,
LT_GT_GT_GT_LT_LT=78,
EQ_GT_GT_GT_LT_LT=79,
GT_GT_GT_GT_LT_LT=80,
LT_LT_LT_LT_EQ_LT=81,
EQ_LT_LT_LT_EQ_LT=82,
GT_LT_LT_LT_EQ_LT=83,
LT_EQ_LT_LT_EQ_LT=84,
EQ_EQ_LT_LT_EQ_LT=85,
GT_EQ_LT_LT_EQ_LT=86,
LT_GT_LT_LT_EQ_LT=87,
EQ_GT_LT_LT_EQ_LT=88,
GT_GT_LT_LT_EQ_LT=89,
LT_LT_EQ_LT_EQ_LT=90,
EQ_LT_EQ_LT_EQ_LT=91,
GT_LT_EQ_LT_EQ_LT=92,
LT_EQ_EQ_LT_EQ_LT=93,
EQ_EQ_EQ_LT_EQ_LT=94,
GT_EQ_EQ_LT_EQ_LT=95,
LT_GT_EQ_LT_EQ_LT=96,
EQ_GT_EQ_LT_EQ_LT=97,
GT_GT_EQ_LT_EQ_LT=98,
LT_LT_GT_LT_EQ_LT=99,
EQ_LT_GT_LT_EQ_LT=100,
GT_LT_GT_LT_EQ_LT=101,
LT_EQ_GT_LT_EQ_LT=102,
EQ_EQ_GT_LT_EQ_LT=103,
GT_EQ_GT_LT_EQ_LT=104,
LT_GT_GT_LT_EQ_LT=105,
EQ_GT_GT_LT_EQ_LT=106,
GT_GT_GT_LT_EQ_LT=107,
LT_LT_LT_EQ_EQ_LT=108,
EQ_LT_LT_EQ_EQ_LT=109,
GT_LT_LT_EQ_EQ_LT=110,
LT_EQ_LT_EQ_EQ_LT=111,
EQ_EQ_LT_EQ_EQ_LT=112,
GT_EQ_LT_EQ_EQ_LT=113,
LT_GT_LT_EQ_EQ_LT=114,
EQ_GT_LT_EQ_EQ_LT=115,
GT_GT_LT_EQ_EQ_LT=116,
LT_LT_EQ_EQ_EQ_LT=117,
EQ_LT_EQ_EQ_EQ_LT=118,
GT_LT_EQ_EQ_EQ_LT=119,
LT_EQ_EQ_EQ_EQ_LT=120,
EQ_EQ_EQ_EQ_EQ_LT=121,
GT_EQ_EQ_EQ_EQ_LT=122,
LT_GT_EQ_EQ_EQ_LT=123,
EQ_GT_EQ_EQ_EQ_LT=124,
GT_GT_EQ_EQ_EQ_LT=125,
LT_LT_GT_EQ_EQ_LT=126,
EQ_LT_GT_EQ_EQ_LT=127,
GT_LT_GT_EQ_EQ_LT=128,
LT_EQ_GT_EQ_EQ_LT=129,
EQ_EQ_GT_EQ_EQ_LT=130,
GT_EQ_GT_EQ_EQ_LT=131,
LT_GT_GT_EQ_EQ_LT=132,
EQ_GT_GT_EQ_EQ_LT=133,
GT_GT_GT_EQ_EQ_LT=134,
LT_LT_LT_GT_EQ_LT=135,
EQ_LT_LT_GT_EQ_LT=136,
GT_LT_LT_GT_EQ_LT=137,
LT_EQ_LT_GT_EQ_LT=138,
EQ_EQ_LT_GT_EQ_LT=139,
GT_EQ_LT_GT_EQ_LT=140,
LT_GT_LT_GT_EQ_LT=141,
EQ_GT_LT_GT_EQ_LT=142,
GT_GT_LT_GT_EQ_LT=143,
LT_LT_EQ_GT_EQ_LT=144,
EQ_LT_EQ_GT_EQ_LT=145,
GT_LT_EQ_GT_EQ_LT=146,
LT_EQ_EQ_GT_EQ_LT=147,
EQ_EQ_EQ_GT_EQ_LT=148,
GT_EQ_EQ_GT_EQ_LT=149,
LT_GT_EQ_GT_EQ_LT=150,
EQ_GT_EQ_GT_EQ_LT=151,
GT_GT_EQ_GT_EQ_LT=152,
LT_LT_GT_GT_EQ_LT=153,
EQ_LT_GT_GT_EQ_LT=154,
GT_LT_GT_GT_EQ_LT=155,
LT_EQ_GT_GT_EQ_LT=156,
EQ_EQ_GT_GT_EQ_LT=157,
GT_EQ_GT_GT_EQ_LT=158,
LT_GT_GT_GT_EQ_LT=159,
EQ_GT_GT_GT_EQ_LT=160,
GT_GT_GT_GT_EQ_LT=161,
LT_LT_LT_LT_GT_LT=162,
EQ_LT_LT_LT_GT_LT=163,
GT_LT_LT_LT_GT_LT=164,
LT_EQ_LT_LT_GT_LT=165,
EQ_EQ_LT_LT_GT_LT=166,
GT_EQ_LT_LT_GT_LT=167,
LT_GT_LT_LT_GT_LT=168,
EQ_GT_LT_LT_GT_LT=169,
GT_GT_LT_LT_GT_LT=170,
LT_LT_EQ_LT_GT_LT=171,
EQ_LT_EQ_LT_GT_LT=172,
GT_LT_EQ_LT_GT_LT=173,
LT_EQ_EQ_LT_GT_LT=174,
EQ_EQ_EQ_LT_GT_LT=175,
GT_EQ_EQ_LT_GT_LT=176,
LT_GT_EQ_LT_GT_LT=177,
EQ_GT_EQ_LT_GT_LT=178,
GT_GT_EQ_LT_GT_LT=179,
LT_LT_GT_LT_GT_LT=180,
EQ_LT_GT_LT_GT_LT=181,
GT_LT_GT_LT_GT_LT=182,
LT_EQ_GT_LT_GT_LT=183,
EQ_EQ_GT_LT_GT_LT=184,
GT_EQ_GT_LT_GT_LT=185,
LT_GT_GT_LT_GT_LT=186,
EQ_GT_GT_LT_GT_LT=187,
GT_GT_GT_LT_GT_LT=188,
LT_LT_LT_EQ_GT_LT=189,
EQ_LT_LT_EQ_GT_LT=190,
GT_LT_LT_EQ_GT_LT=191,
LT_EQ_LT_EQ_GT_LT=192,
EQ_EQ_LT_EQ_GT_LT=193,
GT_EQ_LT_EQ_GT_LT=194,
LT_GT_LT_EQ_GT_LT=195,
EQ_GT_LT_EQ_GT_LT=196,
GT_GT_LT_EQ_GT_LT=197,
LT_LT_EQ_EQ_GT_LT=198,
EQ_LT_EQ_EQ_GT_LT=199,
GT_LT_EQ_EQ_GT_LT=200,
LT_EQ_EQ_EQ_GT_LT=201,
EQ_EQ_EQ_EQ_GT_LT=202,
GT_EQ_EQ_EQ_GT_LT=203,
LT_GT_EQ_EQ_GT_LT=204,
EQ_GT_EQ_EQ_GT_LT=205,
GT_GT_EQ_EQ_GT_LT=206,
LT_LT_GT_EQ_GT_LT=207,
EQ_LT_GT_EQ_GT_LT=208,
GT_LT_GT_EQ_GT_LT=209,
LT_EQ_GT_EQ_GT_LT=210,
EQ_EQ_GT_EQ_GT_LT=211,
GT_EQ_GT_EQ_GT_LT=212,
LT_GT_GT_EQ_GT_LT=213,
EQ_GT_GT_EQ_GT_LT=214,
GT_GT_GT_EQ_GT_LT=215,
LT_LT_LT_GT_GT_LT=216,
EQ_LT_LT_GT_GT_LT=217,
GT_LT_LT_GT_GT_LT=218,
LT_EQ_LT_GT_GT_LT=219,
EQ_EQ_LT_GT_GT_LT=220,
GT_EQ_LT_GT_GT_LT=221,
LT_GT_LT_GT_GT_LT=222,
EQ_GT_LT_GT_GT_LT=223,
GT_GT_LT_GT_GT_LT=224,
LT_LT_EQ_GT_GT_LT=225,
EQ_LT_EQ_GT_GT_LT=226,
GT_LT_EQ_GT_GT_LT=227,
LT_EQ_EQ_GT_GT_LT=228,
EQ_EQ_EQ_GT_GT_LT=229,
GT_EQ_EQ_GT_GT_LT=230,
LT_GT_EQ_GT_GT_LT=231,
EQ_GT_EQ_GT_GT_LT=232,
GT_GT_EQ_GT_GT_LT=233,
LT_LT_GT_GT_GT_LT=234,
EQ_LT_GT_GT_GT_LT=235,
GT_LT_GT_GT_GT_LT=236,
LT_EQ_GT_GT_GT_LT=237,
EQ_EQ_GT_GT_GT_LT=238,
GT_EQ_GT_GT_GT_LT=239,
LT_GT_GT_GT_GT_LT=240,
EQ_GT_GT_GT_GT_LT=241,
GT_GT_GT_GT_GT_LT=242,
LT_LT_LT_LT_LT_EQ=243,
EQ_LT_LT_LT_LT_EQ=244,
GT_LT_LT_LT_LT_EQ=245,
LT_EQ_LT_LT_LT_EQ=246,
EQ_EQ_LT_LT_LT_EQ=247,
GT_EQ_LT_LT_LT_EQ=248,
LT_GT_LT_LT_LT_EQ=249,
EQ_GT_LT_LT_LT_EQ=250,
GT_GT_LT_LT_LT_EQ=251,
LT_LT_EQ_LT_LT_EQ=252,
EQ_LT_EQ_LT_LT_EQ=253,
GT_LT_EQ_LT_LT_EQ=254,
LT_EQ_EQ_LT_LT_EQ=255,
EQ_EQ_EQ_LT_LT_EQ=256,
GT_EQ_EQ_LT_LT_EQ=257,
LT_GT_EQ_LT_LT_EQ=258,
EQ_GT_EQ_LT_LT_EQ=259,
GT_GT_EQ_LT_LT_EQ=260,
LT_LT_GT_LT_LT_EQ=261,
EQ_LT_GT_LT_LT_EQ=262,
GT_LT_GT_LT_LT_EQ=263,
LT_EQ_GT_LT_LT_EQ=264,
EQ_EQ_GT_LT_LT_EQ=265,
GT_EQ_GT_LT_LT_EQ=266,
LT_GT_GT_LT_LT_EQ=267,
EQ_GT_GT_LT_LT_EQ=268,
GT_GT_GT_LT_LT_EQ=269,
LT_LT_LT_EQ_LT_EQ=270,
EQ_LT_LT_EQ_LT_EQ=271,
GT_LT_LT_EQ_LT_EQ=272,
LT_EQ_LT_EQ_LT_EQ=273,
EQ_EQ_LT_EQ_LT_EQ=274,
GT_EQ_LT_EQ_LT_EQ=275,
LT_GT_LT_EQ_LT_EQ=276,
EQ_GT_LT_EQ_LT_EQ=277,
GT_GT_LT_EQ_LT_EQ=278,
LT_LT_EQ_EQ_LT_EQ=279,
EQ_LT_EQ_EQ_LT_EQ=280,
GT_LT_EQ_EQ_LT_EQ=281,
LT_EQ_EQ_EQ_LT_EQ=282,
EQ_EQ_EQ_EQ_LT_EQ=283,
GT_EQ_EQ_EQ_LT_EQ=284,
LT_GT_EQ_EQ_LT_EQ=285,
EQ_GT_EQ_EQ_LT_EQ=286,
GT_GT_EQ_EQ_LT_EQ=287,
LT_LT_GT_EQ_LT_EQ=288,
EQ_LT_GT_EQ_LT_EQ=289,
GT_LT_GT_EQ_LT_EQ=290,
LT_EQ_GT_EQ_LT_EQ=291,
EQ_EQ_GT_EQ_LT_EQ=292,
GT_EQ_GT_EQ_LT_EQ=293,
LT_GT_GT_EQ_LT_EQ=294,
EQ_GT_GT_EQ_LT_EQ=295,
GT_GT_GT_EQ_LT_EQ=296,
LT_LT_LT_GT_LT_EQ=297,
EQ_LT_LT_GT_LT_EQ=298,
GT_LT_LT_GT_LT_EQ=299,
LT_EQ_LT_GT_LT_EQ=300,
EQ_EQ_LT_GT_LT_EQ=301,
GT_EQ_LT_GT_LT_EQ=302,
LT_GT_LT_GT_LT_EQ=303,
EQ_GT_LT_GT_LT_EQ=304,
GT_GT_LT_GT_LT_EQ=305,
LT_LT_EQ_GT_LT_EQ=306,
EQ_LT_EQ_GT_LT_EQ=307,
GT_LT_EQ_GT_LT_EQ=308,
LT_EQ_EQ_GT_LT_EQ=309,
EQ_EQ_EQ_GT_LT_EQ=310,
GT_EQ_EQ_GT_LT_EQ=311,
LT_GT_EQ_GT_LT_EQ=312,
EQ_GT_EQ_GT_LT_EQ=313,
GT_GT_EQ_GT_LT_EQ=314,
LT_LT_GT_GT_LT_EQ=315,
EQ_LT_GT_GT_LT_EQ=316,
GT_LT_GT_GT_LT_EQ=317,
LT_EQ_GT_GT_LT_EQ=318,
EQ_EQ_GT_GT_LT_EQ=319,
GT_EQ_GT_GT_LT_EQ=320,
LT_GT_GT_GT_LT_EQ=321,
EQ_GT_GT_GT_LT_EQ=322,
GT_GT_GT_GT_LT_EQ=323,
LT_LT_LT_LT_EQ_EQ=324,
EQ_LT_LT_LT_EQ_EQ=325,
GT_LT_LT_LT_EQ_EQ=326,
LT_EQ_LT_LT_EQ_EQ=327,
EQ_EQ_LT_LT_EQ_EQ=328,
GT_EQ_LT_LT_EQ_EQ=329,
LT_GT_LT_LT_EQ_EQ=330,
EQ_GT_LT_LT_EQ_EQ=331,
GT_GT_LT_LT_EQ_EQ=332,
LT_LT_EQ_LT_EQ_EQ=333,
EQ_LT_EQ_LT_EQ_EQ=334,
GT_LT_EQ_LT_EQ_EQ=335,
LT_EQ_EQ_LT_EQ_EQ=336,
EQ_EQ_EQ_LT_EQ_EQ=337,
GT_EQ_EQ_LT_EQ_EQ=338,
LT_GT_EQ_LT_EQ_EQ=339,
EQ_GT_EQ_LT_EQ_EQ=340,
GT_GT_EQ_LT_EQ_EQ=341,
LT_LT_GT_LT_EQ_EQ=342,
EQ_LT_GT_LT_EQ_EQ=343,
GT_LT_GT_LT_EQ_EQ=344,
LT_EQ_GT_LT_EQ_EQ=345,
EQ_EQ_GT_LT_EQ_EQ=346,
GT_EQ_GT_LT_EQ_EQ=347,
LT_GT_GT_LT_EQ_EQ=348,
EQ_GT_GT_LT_EQ_EQ=349,
GT_GT_GT_LT_EQ_EQ=350,
LT_LT_LT_EQ_EQ_EQ=351,
EQ_LT_LT_EQ_EQ_EQ=352,
GT_LT_LT_EQ_EQ_EQ=353,
LT_EQ_LT_EQ_EQ_EQ=354,
EQ_EQ_LT_EQ_EQ_EQ=355,
GT_EQ_LT_EQ_EQ_EQ=356,
LT_GT_LT_EQ_EQ_EQ=357,
EQ_GT_LT_EQ_EQ_EQ=358,
GT_GT_LT_EQ_EQ_EQ=359,
LT_LT_EQ_EQ_EQ_EQ=360,
EQ_LT_EQ_EQ_EQ_EQ=361,
GT_LT_EQ_EQ_EQ_EQ=362,
LT_EQ_EQ_EQ_EQ_EQ=363,
EQ_EQ_EQ_EQ_EQ_EQ=364,
GT_EQ_EQ_EQ_EQ_EQ=365,
LT_GT_EQ_EQ_EQ_EQ=366,
EQ_GT_EQ_EQ_EQ_EQ=367,
GT_GT_EQ_EQ_EQ_EQ=368,
LT_LT_GT_EQ_EQ_EQ=369,
EQ_LT_GT_EQ_EQ_EQ=370,
GT_LT_GT_EQ_EQ_EQ=371,
LT_EQ_GT_EQ_EQ_EQ=372,
EQ_EQ_GT_EQ_EQ_EQ=373,
GT_EQ_GT_EQ_EQ_EQ=374,
LT_GT_GT_EQ_EQ_EQ=375,
EQ_GT_GT_EQ_EQ_EQ=376,
GT_GT_GT_EQ_EQ_EQ=377,
LT_LT_LT_GT_EQ_EQ=378,
EQ_LT_LT_GT_EQ_EQ=379,
GT_LT_LT_GT_EQ_EQ=380,
LT_EQ_LT_GT_EQ_EQ=381,
EQ_EQ_LT_GT_EQ_EQ=382,
GT_EQ_LT_GT_EQ_EQ=383,
LT_GT_LT_GT_EQ_EQ=384,
EQ_GT_LT_GT_EQ_EQ=385,
GT_GT_LT_GT_EQ_EQ=386,
LT_LT_EQ_GT_EQ_EQ=387,
EQ_LT_EQ_GT_EQ_EQ=388,
GT_LT_EQ_GT_EQ_EQ=389,
LT_EQ_EQ_GT_EQ_EQ=390,
EQ_EQ_EQ_GT_EQ_EQ=391,
GT_EQ_EQ_GT_EQ_EQ=392,
LT_GT_EQ_GT_EQ_EQ=393,
EQ_GT_EQ_GT_EQ_EQ=394,
GT_GT_EQ_GT_EQ_EQ=395,
LT_LT_GT_GT_EQ_EQ=396,
EQ_LT_GT_GT_EQ_EQ=397,
GT_LT_GT_GT_EQ_EQ=398,
LT_EQ_GT_GT_EQ_EQ=399,
EQ_EQ_GT_GT_EQ_EQ=400,
GT_EQ_GT_GT_EQ_EQ=401,
LT_GT_GT_GT_EQ_EQ=402,
EQ_GT_GT_GT_EQ_EQ=403,
GT_GT_GT_GT_EQ_EQ=404,
LT_LT_LT_LT_GT_EQ=405,
EQ_LT_LT_LT_GT_EQ=406,
GT_LT_LT_LT_GT_EQ=407,
LT_EQ_LT_LT_GT_EQ=408,
EQ_EQ_LT_LT_GT_EQ=409,
GT_EQ_LT_LT_GT_EQ=410,
LT_GT_LT_LT_GT_EQ=411,
EQ_GT_LT_LT_GT_EQ=412,
GT_GT_LT_LT_GT_EQ=413,
LT_LT_EQ_LT_GT_EQ=414,
EQ_LT_EQ_LT_GT_EQ=415,
GT_LT_EQ_LT_GT_EQ=416,
LT_EQ_EQ_LT_GT_EQ=417,
EQ_EQ_EQ_LT_GT_EQ=418,
GT_EQ_EQ_LT_GT_EQ=419,
LT_GT_EQ_LT_GT_EQ=420,
EQ_GT_EQ_LT_GT_EQ=421,
GT_GT_EQ_LT_GT_EQ=422,
LT_LT_GT_LT_GT_EQ=423,
EQ_LT_GT_LT_GT_EQ=424,
GT_LT_GT_LT_GT_EQ=425,
LT_EQ_GT_LT_GT_EQ=426,
EQ_EQ_GT_LT_GT_EQ=427,
GT_EQ_GT_LT_GT_EQ=428,
LT_GT_GT_LT_GT_EQ=429,
EQ_GT_GT_LT_GT_EQ=430,
GT_GT_GT_LT_GT_EQ=431,
LT_LT_LT_EQ_GT_EQ=432,
EQ_LT_LT_EQ_GT_EQ=433,
GT_LT_LT_EQ_GT_EQ=434,
LT_EQ_LT_EQ_GT_EQ=435,
EQ_EQ_LT_EQ_GT_EQ=436,
GT_EQ_LT_EQ_GT_EQ=437,
LT_GT_LT_EQ_GT_EQ=438,
EQ_GT_LT_EQ_GT_EQ=439,
GT_GT_LT_EQ_GT_EQ=440,
LT_LT_EQ_EQ_GT_EQ=441,
EQ_LT_EQ_EQ_GT_EQ=442,
GT_LT_EQ_EQ_GT_EQ=443,
LT_EQ_EQ_EQ_GT_EQ=444,
EQ_EQ_EQ_EQ_GT_EQ=445,
GT_EQ_EQ_EQ_GT_EQ=446,
LT_GT_EQ_EQ_GT_EQ=447,
EQ_GT_EQ_EQ_GT_EQ=448,
GT_GT_EQ_EQ_GT_EQ=449,
LT_LT_GT_EQ_GT_EQ=450,
EQ_LT_GT_EQ_GT_EQ=451,
GT_LT_GT_EQ_GT_EQ=452,
LT_EQ_GT_EQ_GT_EQ=453,
EQ_EQ_GT_EQ_GT_EQ=454,
GT_EQ_GT_EQ_GT_EQ=455,
LT_GT_GT_EQ_GT_EQ=456,
EQ_GT_GT_EQ_GT_EQ=457,
GT_GT_GT_EQ_GT_EQ=458,
LT_LT_LT_GT_GT_EQ=459,
EQ_LT_LT_GT_GT_EQ=460,
GT_LT_LT_GT_GT_EQ=461,
LT_EQ_LT_GT_GT_EQ=462,
EQ_EQ_LT_GT_GT_EQ=463,
GT_EQ_LT_GT_GT_EQ=464,
LT_GT_LT_GT_GT_EQ=465,
EQ_GT_LT_GT_GT_EQ=466,
GT_GT_LT_GT_GT_EQ=467,
LT_LT_EQ_GT_GT_EQ=468,
EQ_LT_EQ_GT_GT_EQ=469,
GT_LT_EQ_GT_GT_EQ=470,
LT_EQ_EQ_GT_GT_EQ=471,
EQ_EQ_EQ_GT_GT_EQ=472,
GT_EQ_EQ_GT_GT_EQ=473,
LT_GT_EQ_GT_GT_EQ=474,
EQ_GT_EQ_GT_GT_EQ=475,
GT_GT_EQ_GT_GT_EQ=476,
LT_LT_GT_GT_GT_EQ=477,
EQ_LT_GT_GT_GT_EQ=478,
GT_LT_GT_GT_GT_EQ=479,
LT_EQ_GT_GT_GT_EQ=480,
EQ_EQ_GT_GT_GT_EQ=481,
GT_EQ_GT_GT_GT_EQ=482,
LT_GT_GT_GT_GT_EQ=483,
EQ_GT_GT_GT_GT_EQ=484,
GT_GT_GT_GT_GT_EQ=485,
LT_LT_LT_LT_LT_GT=486,
EQ_LT_LT_LT_LT_GT=487,
GT_LT_LT_LT_LT_GT=488,
LT_EQ_LT_LT_LT_GT=489,
EQ_EQ_LT_LT_LT_GT=490,
GT_EQ_LT_LT_LT_GT=491,
LT_GT_LT_LT_LT_GT=492,
EQ_GT_LT_LT_LT_GT=493,
GT_GT_LT_LT_LT_GT=494,
LT_LT_EQ_LT_LT_GT=495,
EQ_LT_EQ_LT_LT_GT=496,
GT_LT_EQ_LT_LT_GT=497,
LT_EQ_EQ_LT_LT_GT=498,
EQ_EQ_EQ_LT_LT_GT=499,
GT_EQ_EQ_LT_LT_GT=500,
LT_GT_EQ_LT_LT_GT=501,
EQ_GT_EQ_LT_LT_GT=502,
GT_GT_EQ_LT_LT_GT=503,
LT_LT_GT_LT_LT_GT=504,
EQ_LT_GT_LT_LT_GT=505,
GT_LT_GT_LT_LT_GT=506,
LT_EQ_GT_LT_LT_GT=507,
EQ_EQ_GT_LT_LT_GT=508,
GT_EQ_GT_LT_LT_GT=509,
LT_GT_GT_LT_LT_GT=510,
EQ_GT_GT_LT_LT_GT=511,
GT_GT_GT_LT_LT_GT=512,
LT_LT_LT_EQ_LT_GT=513,
EQ_LT_LT_EQ_LT_GT=514,
GT_LT_LT_EQ_LT_GT=515,
LT_EQ_LT_EQ_LT_GT=516,
EQ_EQ_LT_EQ_LT_GT=517,
GT_EQ_LT_EQ_LT_GT=518,
LT_GT_LT_EQ_LT_GT=519,
EQ_GT_LT_EQ_LT_GT=520,
GT_GT_LT_EQ_LT_GT=521,
LT_LT_EQ_EQ_LT_GT=522,
EQ_LT_EQ_EQ_LT_GT=523,
GT_LT_EQ_EQ_LT_GT=524,
LT_EQ_EQ_EQ_LT_GT=525,
EQ_EQ_EQ_EQ_LT_GT=526,
GT_EQ_EQ_EQ_LT_GT=527,
LT_GT_EQ_EQ_LT_GT=528,
EQ_GT_EQ_EQ_LT_GT=529,
GT_GT_EQ_EQ_LT_GT=530,
LT_LT_GT_EQ_LT_GT=531,
EQ_LT_GT_EQ_LT_GT=532,
GT_LT_GT_EQ_LT_GT=533,
LT_EQ_GT_EQ_LT_GT=534,
EQ_EQ_GT_EQ_LT_GT=535,
GT_EQ_GT_EQ_LT_GT=536,
LT_GT_GT_EQ_LT_GT=537,
EQ_GT_GT_EQ_LT_GT=538,
GT_GT_GT_EQ_LT_GT=539,
LT_LT_LT_GT_LT_GT=540,
EQ_LT_LT_GT_LT_GT=541,
GT_LT_LT_GT_LT_GT=542,
LT_EQ_LT_GT_LT_GT=543,
EQ_EQ_LT_GT_LT_GT=544,
GT_EQ_LT_GT_LT_GT=545,
LT_GT_LT_GT_LT_GT=546,
EQ_GT_LT_GT_LT_GT=547,
GT_GT_LT_GT_LT_GT=548,
LT_LT_EQ_GT_LT_GT=549,
EQ_LT_EQ_GT_LT_GT=550,
GT_LT_EQ_GT_LT_GT=551,
LT_EQ_EQ_GT_LT_GT=552,
EQ_EQ_EQ_GT_LT_GT=553,
GT_EQ_EQ_GT_LT_GT=554,
LT_GT_EQ_GT_LT_GT=555,
EQ_GT_EQ_GT_LT_GT=556,
GT_GT_EQ_GT_LT_GT=557,
LT_LT_GT_GT_LT_GT=558,
EQ_LT_GT_GT_LT_GT=559,
GT_LT_GT_GT_LT_GT=560,
LT_EQ_GT_GT_LT_GT=561,
EQ_EQ_GT_GT_LT_GT=562,
GT_EQ_GT_GT_LT_GT=563,
LT_GT_GT_GT_LT_GT=564,
EQ_GT_GT_GT_LT_GT=565,
GT_GT_GT_GT_LT_GT=566,
LT_LT_LT_LT_EQ_GT=567,
EQ_LT_LT_LT_EQ_GT=568,
GT_LT_LT_LT_EQ_GT=569,
LT_EQ_LT_LT_EQ_GT=570,
EQ_EQ_LT_LT_EQ_GT=571,
GT_EQ_LT_LT_EQ_GT=572,
LT_GT_LT_LT_EQ_GT=573,
EQ_GT_LT_LT_EQ_GT=574,
GT_GT_LT_LT_EQ_GT=575,
LT_LT_EQ_LT_EQ_GT=576,
EQ_LT_EQ_LT_EQ_GT=577,
GT_LT_EQ_LT_EQ_GT=578,
LT_EQ_EQ_LT_EQ_GT=579,
EQ_EQ_EQ_LT_EQ_GT=580,
GT_EQ_EQ_LT_EQ_GT=581,
LT_GT_EQ_LT_EQ_GT=582,
EQ_GT_EQ_LT_EQ_GT=583,
GT_GT_EQ_LT_EQ_GT=584,
LT_LT_GT_LT_EQ_GT=585,
EQ_LT_GT_LT_EQ_GT=586,
GT_LT_GT_LT_EQ_GT=587,
LT_EQ_GT_LT_EQ_GT=588,
EQ_EQ_GT_LT_EQ_GT=589,
GT_EQ_GT_LT_EQ_GT=590,
LT_GT_GT_LT_EQ_GT=591,
EQ_GT_GT_LT_EQ_GT=592,
GT_GT_GT_LT_EQ_GT=593,
LT_LT_LT_EQ_EQ_GT=594,
EQ_LT_LT_EQ_EQ_GT=595,
GT_LT_LT_EQ_EQ_GT=596,
LT_EQ_LT_EQ_EQ_GT=597,
EQ_EQ_LT_EQ_EQ_GT=598,
GT_EQ_LT_EQ_EQ_GT=599,
LT_GT_LT_EQ_EQ_GT=600,
EQ_GT_LT_EQ_EQ_GT=601,
GT_GT_LT_EQ_EQ_GT=602,
LT_LT_EQ_EQ_EQ_GT=603,
EQ_LT_EQ_EQ_EQ_GT=604,
GT_LT_EQ_EQ_EQ_GT=605,
LT_EQ_EQ_EQ_EQ_GT=606,
EQ_EQ_EQ_EQ_EQ_GT=607,
GT_EQ_EQ_EQ_EQ_GT=608,
LT_GT_EQ_EQ_EQ_GT=609,
EQ_GT_EQ_EQ_EQ_GT=610,
GT_GT_EQ_EQ_EQ_GT=611,
LT_LT_GT_EQ_EQ_GT=612,
EQ_LT_GT_EQ_EQ_GT=613,
GT_LT_GT_EQ_EQ_GT=614,
LT_EQ_GT_EQ_EQ_GT=615,
EQ_EQ_GT_EQ_EQ_GT=616,
GT_EQ_GT_EQ_EQ_GT=617,
LT_GT_GT_EQ_EQ_GT=618,
EQ_GT_GT_EQ_EQ_GT=619,
GT_GT_GT_EQ_EQ_GT=620,
LT_LT_LT_GT_EQ_GT=621,
EQ_LT_LT_GT_EQ_GT=622,
GT_LT_LT_GT_EQ_GT=623,
LT_EQ_LT_GT_EQ_GT=624,
EQ_EQ_LT_GT_EQ_GT=625,
GT_EQ_LT_GT_EQ_GT=626,
LT_GT_LT_GT_EQ_GT=627,
EQ_GT_LT_GT_EQ_GT=628,
GT_GT_LT_GT_EQ_GT=629,
LT_LT_EQ_GT_EQ_GT=630,
EQ_LT_EQ_GT_EQ_GT=631,
GT_LT_EQ_GT_EQ_GT=632,
LT_EQ_EQ_GT_EQ_GT=633,
EQ_EQ_EQ_GT_EQ_GT=634,
GT_EQ_EQ_GT_EQ_GT=635,
LT_GT_EQ_GT_EQ_GT=636,
EQ_GT_EQ_GT_EQ_GT=637,
GT_GT_EQ_GT_EQ_GT=638,
LT_LT_GT_GT_EQ_GT=639,
EQ_LT_GT_GT_EQ_GT=640,
GT_LT_GT_GT_EQ_GT=641,
LT_EQ_GT_GT_EQ_GT=642,
EQ_EQ_GT_GT_EQ_GT=643,
GT_EQ_GT_GT_EQ_GT=644,
LT_GT_GT_GT_EQ_GT=645,
EQ_GT_GT_GT_EQ_GT=646,
GT_GT_GT_GT_EQ_GT=647,
LT_LT_LT_LT_GT_GT=648,
EQ_LT_LT_LT_GT_GT=649,
GT_LT_LT_LT_GT_GT=650,
LT_EQ_LT_LT_GT_GT=651,
EQ_EQ_LT_LT_GT_GT=652,
GT_EQ_LT_LT_GT_GT=653,
LT_GT_LT_LT_GT_GT=654,
EQ_GT_LT_LT_GT_GT=655,
GT_GT_LT_LT_GT_GT=656,
LT_LT_EQ_LT_GT_GT=657,
EQ_LT_EQ_LT_GT_GT=658,
GT_LT_EQ_LT_GT_GT=659,
LT_EQ_EQ_LT_GT_GT=660,
EQ_EQ_EQ_LT_GT_GT=661,
GT_EQ_EQ_LT_GT_GT=662,
LT_GT_EQ_LT_GT_GT=663,
EQ_GT_EQ_LT_GT_GT=664,
GT_GT_EQ_LT_GT_GT=665,
LT_LT_GT_LT_GT_GT=666,
EQ_LT_GT_LT_GT_GT=667,
GT_LT_GT_LT_GT_GT=668,
LT_EQ_GT_LT_GT_GT=669,
EQ_EQ_GT_LT_GT_GT=670,
GT_EQ_GT_LT_GT_GT=671,
LT_GT_GT_LT_GT_GT=672,
EQ_GT_GT_LT_GT_GT=673,
GT_GT_GT_LT_GT_GT=674,
LT_LT_LT_EQ_GT_GT=675,
EQ_LT_LT_EQ_GT_GT=676,
GT_LT_LT_EQ_GT_GT=677,
LT_EQ_LT_EQ_GT_GT=678,
EQ_EQ_LT_EQ_GT_GT=679,
GT_EQ_LT_EQ_GT_GT=680,
LT_GT_LT_EQ_GT_GT=681,
EQ_GT_LT_EQ_GT_GT=682,
GT_GT_LT_EQ_GT_GT=683,
LT_LT_EQ_EQ_GT_GT=684,
EQ_LT_EQ_EQ_GT_GT=685,
GT_LT_EQ_EQ_GT_GT=686,
LT_EQ_EQ_EQ_GT_GT=687,
EQ_EQ_EQ_EQ_GT_GT=688,
GT_EQ_EQ_EQ_GT_GT=689,
LT_GT_EQ_EQ_GT_GT=690,
EQ_GT_EQ_EQ_GT_GT=691,
GT_GT_EQ_EQ_GT_GT=692,
LT_LT_GT_EQ_GT_GT=693,
EQ_LT_GT_EQ_GT_GT=694,
GT_LT_GT_EQ_GT_GT=695,
LT_EQ_GT_EQ_GT_GT=696,
EQ_EQ_GT_EQ_GT_GT=697,
GT_EQ_GT_EQ_GT_GT=698,
LT_GT_GT_EQ_GT_GT=699,
EQ_GT_GT_EQ_GT_GT=700,
GT_GT_GT_EQ_GT_GT=701,
LT_LT_LT_GT_GT_GT=702,
EQ_LT_LT_GT_GT_GT=703,
GT_LT_LT_GT_GT_GT=704,
LT_EQ_LT_GT_GT_GT=705,
EQ_EQ_LT_GT_GT_GT=706,
GT_EQ_LT_GT_GT_GT=707,
LT_GT_LT_GT_GT_GT=708,
EQ_GT_LT_GT_GT_GT=709,
GT_GT_LT_GT_GT_GT=710,
LT_LT_EQ_GT_GT_GT=711,
EQ_LT_EQ_GT_GT_GT=712,
GT_LT_EQ_GT_GT_GT=713,
LT_EQ_EQ_GT_GT_GT=714,
EQ_EQ_EQ_GT_GT_GT=715,
GT_EQ_EQ_GT_GT_GT=716,
LT_GT_EQ_GT_GT_GT=717,
EQ_GT_EQ_GT_GT_GT=718,
GT_GT_EQ_GT_GT_GT=719,
LT_LT_GT_GT_GT_GT=720,
EQ_LT_GT_GT_GT_GT=721,
GT_LT_GT_GT_GT_GT=722,
LT_EQ_GT_GT_GT_GT=723,
EQ_EQ_GT_GT_GT_GT=724,
GT_EQ_GT_GT_GT_GT=725,
LT_GT_GT_GT_GT_GT=726,
EQ_GT_GT_GT_GT_GT=727,
GT_GT_GT_GT_GT_GT=728,
//----------------------------
END_OF_ALL_THE_HASHED_CASES
};



struct TriplePoint
{
 float coord[3];
};

struct normalArray
{

    struct TriplePoint point[3];

    float normal[3];
};


int swapResultPoints(unsigned int id ,struct normalArray * result, unsigned int swapIDA,unsigned int swapIDB)
{
  if (swapIDA==swapIDB) { return 1; }

  float tmp=0.0;
  unsigned int i=0;
  for (i=0; i<3; i++)
     {
        tmp = result[id].point[swapIDA].coord[i];
        result[id].point[swapIDA].coord[i] = result[id].point[swapIDB].coord[i];
        result[id].point[swapIDB].coord[i] = tmp;
     }
  return 1;
}


int ensureClockwise(unsigned int id , struct normalArray * result)
{
  unsigned int swapA=0,swapB=1;

  struct TriplePoint legend;
  legend.coord[X]=0.016560; legend.coord[Y]=-0.826509; legend.coord[Z]=-0.562679;


  float retres = dotProduct(result->normal , legend.coord );

  if (retres<0.0) { swapResultPoints(id,result,swapA,swapB); }

  return 1;
}



int ensureClockwiseCrazy(unsigned int id , struct normalArray * result)
{
  unsigned int swapA=0,swapB=0;

  float X21f = result[id].point[Point2].coord[X] - result[id].point[Point1].coord[X];
  float Y21f = result[id].point[Point2].coord[Y] - result[id].point[Point1].coord[Y];
  float Z21f = result[id].point[Point2].coord[Z] - result[id].point[Point1].coord[Z];

  float X23f = result[id].point[Point2].coord[X] - result[id].point[Point3].coord[X];
  float Y23f = result[id].point[Point2].coord[Y] - result[id].point[Point3].coord[Y];
  float Z23f = result[id].point[Point2].coord[Z] - result[id].point[Point3].coord[Z];

  unsigned int X21sign,Y21sign,Z21sign,X23sign,Y23sign,Z23sign;

  if (X21f<0.0) {X21sign=LT; } else if (X21f>0.0) {X21sign=GT; } else { X21sign=EQ; }
  if (Y21f<0.0) {Y21sign=LT; } else if (Y21f>0.0) {Y21sign=GT; } else { Y21sign=EQ; }
  if (Z21f<0.0) {Z21sign=LT; } else if (Z21f>0.0) {Z21sign=GT; } else { Z21sign=EQ; }

  if (X23f<0.0) {X23sign=LT; } else if (X23f>0.0) {X23sign=GT; } else { X23sign=EQ; }
  if (Y23f<0.0) {Y23sign=LT; } else if (Y23f>0.0) {Y23sign=GT; } else { Y23sign=EQ; }
  if (Z23f<0.0) {Z23sign=LT; } else if (Z23f>0.0) {Z23sign=GT; } else { Z23sign=EQ; }


  unsigned int hashedResponse = X21sign + (3 * Y21sign) + (3*3 * Z21sign) + (3*3*3 * X23sign) + (3*3*3*3 * Y23sign) + (3*3*3*3*3 * Z23sign);

  if ( (X21sign==LT) && (Y21sign==LT)  && (Z21sign==LT) &&
       (X23sign==LT) && (Y23sign==LT)  && (Z23sign==LT) )
  {

  }



  swapResultPoints(id,result,swapA,swapB);


  return 1;

}

int automaticPlaneSegmentation(unsigned short * source , unsigned int width , unsigned int height , float offset, struct SegmentationFeaturesDepth * segConf , struct calibration * calib)
{
    fprintf(stderr,"doing automaticPlaneSegmentation() VER\n");
    //double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
    //if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform plane segmentation\n"); return 0; }
    unsigned int boundDistance=10;
    unsigned int x,y,depth;
    unsigned int bestNormal = 0;

    if (ResultNormals==0) { fprintf(stderr,"No Normals allowed cannot do automatic plane segmentation \n"); return 0; }

    struct normalArray result[ResultNormals]={0};
    unsigned int resultScore[ResultNormals]={0};

    unsigned int tries=0;
    int i=0;
    for (i=0; i<ResultNormals; i++)
    {
        fprintf(stderr,"TryNumber %u \n",i);
         result[i].point[0].coord[Z]=0;
         tries=0; depth=0;
         while ( ( (depth!=0) || (tries==0) || (result[i].point[0].coord[Z]==0) ) && (tries<10000) )
         {
          ++tries;
          x=boundDistance+rand()%(width-1-boundDistance);  y=boundDistance+rand()%(height-1-boundDistance); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0)  {
                           transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].point[0].coord[X] , &result[i].point[0].coord[Y] ,  &result[i].point[0].coord[Z]);
                         }
         }

         fprintf(stderr,"Point1(%u,%u) picked with depth %u \n",x,y,depth);

         result[i].point[1].coord[Z]=0;
         tries=0; depth=0;
         while ( ( (depth!=0) || (tries==0) || (result[i].point[1].coord[Z]==0) ) && (tries<10000) )
         {
          ++tries;
          x=boundDistance+rand()%(width-1-boundDistance);  y=boundDistance+rand()%(height-1-boundDistance); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0) {
                          transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].point[1].coord[X] , &result[i].point[1].coord[Y] ,  &result[i].point[1].coord[Z]);
                        }
         }

         fprintf(stderr,"Point2(%u,%u) picked with depth %u \n",x,y,depth);

         result[i].point[2].coord[Z]=0;
         tries=0; depth=0;
         while ( ( (depth!=0) || (tries==0) || (result[i].point[2].coord[Z]==0) ) && (tries<10000) )
         {
          ++tries;
          x=boundDistance+rand()%(width-1-boundDistance);  y=boundDistance+rand()%(height-1-boundDistance); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0) {
                          transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].point[2].coord[X] , &result[i].point[2].coord[Y] ,  &result[i].point[2].coord[Z]);
                        }
         }

         fprintf(stderr,"Point3(%u,%u) picked with depth %u \n",x,y,depth);

         fprintf(stderr,"3 Points are %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n " ,
                         result[i].point[0].coord[X] ,  result[i].point[0].coord[Y] ,  result[i].point[0].coord[Z] ,
                         result[i].point[1].coord[X] ,  result[i].point[1].coord[Y] ,  result[i].point[1].coord[Z] ,
                         result[i].point[2].coord[X] ,  result[i].point[2].coord[Y] ,  result[i].point[2].coord[Z]
                );

         crossProductFrom3Points( result[i].point[0].coord , result[i].point[1].coord  , result[i].point[2].coord  , result[i].normal);

    }



    int z=0;
    for (i=0; i<ResultNormals; i++)
    {
      for (z=0; z<ResultNormals; z++)
      {
          if (z!=i)
          {
             resultScore[i]+=angleOfNormals(result[i].normal,result[z].normal);
          }
      }
    }

    for (i=0; i<ResultNormals; i++)
    {
        ensureClockwise(i , result);
    }


    float bestScore = 121230.0;
    for (i=0; i<ResultNormals; i++)
    {
      if (resultScore[i]<bestScore)
      {
        bestNormal = i;
        bestScore = resultScore[i];
      }
    }


    fprintf(stderr,"Picked result %u with score %0.2f \n",bestNormal , bestScore);

    segConf->enablePlaneSegmentation=1;
    segConf->planeNormalOffset=offset; //<- this is to ensure a good auto segmentation
    for (i=0; i<3; i++)
      {
       segConf->p1[i]=result[bestNormal].point[0].coord[i];
       segConf->p2[i]=result[bestNormal].point[1].coord[i];
       segConf->p3[i]=result[bestNormal].point[2].coord[i];
      }

   fprintf(stderr,"Best Points are \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n " ,
                         result[bestNormal].point[0].coord[0] ,  result[bestNormal].point[0].coord[1] ,  result[bestNormal].point[0].coord[2] ,
                         result[bestNormal].point[1].coord[0] ,  result[bestNormal].point[1].coord[1] ,  result[bestNormal].point[1].coord[2] ,
                         result[bestNormal].point[2].coord[0] ,  result[bestNormal].point[2].coord[1] ,  result[bestNormal].point[2].coord[2]
         );

   fprintf(stderr,"AUTOMATIC SHUTDOWN OF SEGMENTATION SO THAT DOES NOT DESTORY OUTPUT\n");
   segConf->autoPlaneSegmentation=0;

   // free4x4Matrix(&m);
  return 1;
}
