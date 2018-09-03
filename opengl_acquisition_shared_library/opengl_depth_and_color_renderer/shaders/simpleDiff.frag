#version 130

in vec2 UV;
uniform sampler2D diffedTexture; 

in vec4 color;
out  vec4  colorOUT;

void main() 
{ 
    //colorOUT = color;

    vec2 verticalFlip=UV;
    verticalFlip.y = 1.0f - verticalFlip.y;
	colorOUT.xyz = color.xyz - texture( diffedTexture, verticalFlip ).xyz ; 
    colorOUT.w = 0;

    //colorOUT = vec4(0.0,1.0,0.0,1.0);
} 

