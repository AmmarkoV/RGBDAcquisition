#!/usr/bin/python
import httplib2

resp, content = httplib2.Http().request("http://127.0.0.1:8080/rgb.raw")

darkPixels = 0
brightPixels = 0
count = 0
while (count < 640*480*3):
	if ( content[count]<150 ) : darkPixels=darkPixels+1; 
	if ( content[count]>=150 ) : brightPixels=brightPixels+1; 
	count = count + 1

print "Counted " +str(darkPixels)+" dark pixels from total of "+str(count)+" pixels"
print "Counted " +str(brightPixels)+" bright pixels from total of "+str(count)+" pixels"


#from array import array
#a = array("h") # h = signed short, H = unsigned short
