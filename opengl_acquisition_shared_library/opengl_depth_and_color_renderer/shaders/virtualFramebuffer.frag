#version 130

in vec2 UV;

out vec3 color;

uniform vec3 iResolution;
uniform sampler2D renderedTexture; 
uniform float iTime;

void main()
{ 
	color = texture( renderedTexture, UV ).xyz ;
	//color = texture( renderedTexture, UV + 0.005*vec2( sin(iTime+1024.0*UV.x),cos(iTime+768.0*UV.y)) ).xyz ; 
  
}
 
