#version 130

in vec2 UV;
out vec3 color;
 
uniform sampler2D renderedTexture2; 

uniform vec3 iResolution;
uniform float iTime;

void main()
{    
	color = texture( renderedTexture2, UV ).xyz ; 
    
   // color.y = color.x;
   // color.x=0;
   //color.z=1.0;
}
 
