#version 130

uniform sampler2D renderedTexture; 
in vec4 color;
in vec2 UV;

out vec4 colorOUT;

void main() 
{ 
    colorOUT = color;
    //colorOUT = vec4(0.0,1.0,0.0,1.0);
    colorOUT = vec4( texture( renderedTexture, UV ).xyz , 1.0 );
} 

