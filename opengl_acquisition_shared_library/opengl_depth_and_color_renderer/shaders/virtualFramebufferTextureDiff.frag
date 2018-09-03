#version 130

in vec2 UV;
in vec2 UVDiffTexture;

out vec3 color;



uniform int tileSizeX = 16;
uniform int tileSizeY = 16;

uniform vec3 iResolution;
uniform sampler2D renderedTexture; 
uniform sampler2D diffedTexture; 
uniform float iTime;

void main()
{ 
    vec3 renderData = texture( renderedTexture , UV ).xyz ; 

    vec2 verticalFlip=UVDiffTexture;
    verticalFlip.y = 1.0f - verticalFlip.y;
    vec3 diffData = texture( diffedTexture, verticalFlip ).xyz ; 
   
    if ( (renderData.x==0) && (renderData.y==0) && (renderData.z==0) )
     {} else
     {
	  color.x = abs(renderData.x - diffData.x); 
      color.y = abs(renderData.y - diffData.y); 
      color.z = abs(renderData.z - diffData.z);  
     }

    //color = diffData;
	//color = texture( renderedTexture, UV ).xyz ;
	//color = texture( renderedTexture, UV + 0.005*vec2( sin(iTime+1024.0*UV.x),cos(iTime+768.0*UV.y)) ).xyz ; 
}
 
