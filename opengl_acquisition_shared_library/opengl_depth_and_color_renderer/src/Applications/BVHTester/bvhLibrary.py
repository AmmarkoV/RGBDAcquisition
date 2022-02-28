import ctypes
import os
from ctypes import *
from os.path import exists


def loadLibrary(filename):
 if not exists(filename):
     print("Could not find BVH Library, compiling a fresh one..!")
     os.system("./makeLibrary.sh")
 if not exists(filename):
     print("Could not make BVH Library, terminating")
     sys.exit(0)

 libBVH = CDLL(filename)
 #call C function to check connection
 libBVH.connect() 
 libBVH.bvhConverter.restype = c_int
 libBVH.bvhConverter.argtypes = c_int,POINTER(c_char_p)

 return libBVH



libBVH = loadLibrary("./libBVHConverter.so")

startingFlags=list()
startingFlags.append("--test")
startingFlags.append("--test")
startingFlags.append("--printparams")

argv=startingFlags
argc=len(startingFlags)

#>>> args = (c_char_p * 3)(b'abc',b'def',b'ghi')
#>>> dll.main(len(args),args)


libBVH.bvhConverter(ctypes.c_int(argc),ctypes.c_char_p(argv))

