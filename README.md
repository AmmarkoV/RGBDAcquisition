DepthSenseLinux
===============

Various tools for SoftKinetic's DepthSense 325 camera using Linux SDK
Derived from Damian Lyons' (dlyons@fordham.edu) OpenCV mods of SoftKinetics' ConsoleDemo program (2013)

So far, DepthSenseLinux allows you to:
 - visualize camera streams (both color and depth)
 - perform spatial synchronization using UVmap
 - export frames in JPG or PNM (raw)

Two versions available:
 - DepthSenseGrabberCV (requires OpenCV): allows visualization and JPG/PNM export
 - DepthSenseGrabberSO (stand alone): no online visualization and only PNM export
