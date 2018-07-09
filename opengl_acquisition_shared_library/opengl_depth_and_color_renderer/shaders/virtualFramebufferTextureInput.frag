#version 130

in vec2 UV;
out vec3 color;

layout(binding=0) uniform sampler2D renderedTexture; 

uniform vec3 iResolution;
uniform float iTime;

void main()
{   vec2 verticalFlip=UV;
    //verticalFlip.y = 1.0 - verticalFlip.y
	color = texture( renderedTexture, verticalFlip ).xyz ; 
}
 
