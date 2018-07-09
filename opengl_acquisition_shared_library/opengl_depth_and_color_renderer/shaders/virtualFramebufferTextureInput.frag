#version 130

in vec2 UV;
out vec3 color;

uniform sampler2D renderedTexture; 

uniform vec3 iResolution;
uniform float iTime;

void main()
{   
    vec2 verticalFlip=UV;
    verticalFlip.y = 1.0f - verticalFlip.y;
	color = texture( renderedTexture, verticalFlip ).xyz ; 
}
 
