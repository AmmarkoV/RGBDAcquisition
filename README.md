DepthSenseGrabber
===============

Various tools for SoftKinetic DepthSense 325 camera, for Linux and Windows.

So far, DepthSenseGrabber allows you to:
 - visualize camera streams (both color and depth)
 - perform spatial synchronization using UVmap
 - export frames in JPG, PNM (raw), or PCD (point clouds)

Build with CMake 2.8 or higher.
Dependencies: DepthSense SDK, boost 1.46.0, OpenCV 2.4.10 (optional), PCL 1.2 (optional).

In detail:
 - DepthSenseGrabberCore is the direct interface to the camera and provides access to the color and depth maps (with or without spatial synchronisation), as well as the acquired confidence map.
 - DepthSenseGrabberPNM: no real-time visualisation, grabs images and writes them as PNM.
 - DepthSenseGrabberCV (requires OpenCV): allows visualization, JPG/PNM export.
 - DepthSenseGrabberPCL (requires the Point Cloud Library): no online visualization, PCD export.
