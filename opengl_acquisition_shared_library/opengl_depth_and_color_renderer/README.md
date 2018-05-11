![OpenGL Renderer/Simulator](https://raw.githubusercontent.com/AmmarkoV/RGBDAcquisition/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/doc/logo.png)
This is an GNU/Linux oriented OpenGL renderer that can be used as a simple scripting engine to create simulated datasets..!



## Building
------------------------------------------------------------------ 

To compile the library issue :

```
mkdir build 
cd build 
cmake .. 
make 
```

## Running
------------------------------------------------------------------ 
To run you basically feed the Renderer with a scene file and you get back an OpenGL window where you can see what is happening.
Editing the scene file updates results live on the Renderer 


In order to replay the logo image up you can issue 
```
./Renderer --from Scenes/logo.conf
```



```
./Renderer --from Scenes/hardcodedTest.conf
```


Using the key O you can dump an RGB and Depth image to disk

