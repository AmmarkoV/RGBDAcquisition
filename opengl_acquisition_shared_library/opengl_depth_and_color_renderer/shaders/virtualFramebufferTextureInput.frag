#version 130

in vec2 UV;
out vec3 color;
 
uniform sampler2D renderedTexture2; 

uniform vec3 iResolution;
uniform float iTime;

void main()
{   
    vec2 verticalFlip=UV;
    verticalFlip.y = 1.0f - verticalFlip.y;
	color = texture( renderedTexture2, verticalFlip ).xyz ; 
    
   // color.y = color.x;
   // color.x=0;
   color.z=1.0;
}
 
