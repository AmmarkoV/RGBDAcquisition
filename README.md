DepthSenseGrabber
===============

Various tools for SoftKinetic's DepthSense 325 camera using Linux SDK

So far, DepthSenseGrabber allows you to:
 - visualize camera streams (both color and depth)
 - perform spatial synchronization using UVmap
 - export frames in JPG, PNM (raw), or PCD (point clouds)

Three versions available:
 - DepthSenseGrabberCV (requires OpenCV): allows visualization, JPG/PNM export
 - DepthSenseGrabberPCL (requires the Point Cloud Library): no online visualization, PCD export
 - DepthSenseGrabberSO (stand alone): no online visualization, PNM export
