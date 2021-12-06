#version 130

uniform sampler2D renderedTexture; 

uniform float useTexture = 0;

in vec4 color;
in vec2 UV;

out vec4 colorOUT;

void main() 
{ 
    //colorOUT = color;
    //colorOUT = vec4(0.5,0.5,0.5,1.0);
    //colorOUT = texture(renderedTexture,UV);

    vec4 vertColor   = color * (1.0 - useTexture);
    vec4 vertTexture = texture(renderedTexture,UV);
    colorOUT = vertColor + vertTexture;
} 

