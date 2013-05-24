#!/usr/bin/python
import httplib2

resp, rgb = httplib2.Http().request("http://127.0.0.1:8080/rgb.raw")
resp, depth = httplib2.Http().request("http://127.0.0.1:8080/depth.raw")
resp, forward = httplib2.Http().request("http://127.0.0.1:8080/control.html?snap=1")

darkPixels = 0
brightPixels = 0
count = 0
while (count < 640*480*3):
	if ( rgb[count]<150 ) :
		darkPixels=darkPixels+1 
		
	elif ( rgb[count]>=150 ) :
		brightPixels=brightPixels+1; 
	count = count + 1

print "Counted " +str(darkPixels)+" dark pixels from total of "+str(count)+" pixels"
print "Counted " +str(brightPixels)+" bright pixels from total of "+str(count)+" pixels"


closePixels = 0
farPixels = 0
count = 0
while (count < 640*480):
	if ( depth[count]<150 ) :
		farPixels=farPixels+1 
		
	elif ( depth[count]>=150 ) :
		closePixels=closePixels+1; 
	count = count + 1

print "Counted " +str(closePixels)+" closePixels pixels from total of "+str(count)+" pixels"
print "Counted " +str(farPixels)+" farPixels pixels from total of "+str(count)+" pixels"




#from array import array
#a = array("h") # h = signed short, H = unsigned short
