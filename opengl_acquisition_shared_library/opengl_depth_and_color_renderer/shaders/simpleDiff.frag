#version 130

in vec2 UV;
uniform sampler2D diffedTexture; 

in vec4 color;
out  vec4  colorOUT;

void main() 
{ 
    vec2 verticalFlip=UV;
    verticalFlip.y = 1.0f - verticalFlip.y;

    vec3 textureData = texture( diffedTexture, verticalFlip ).xyz ; 
	colorOUT.x = abs(color.x - textureData.x); 
    colorOUT.y = abs(color.y - textureData.y); 
    colorOUT.z = abs(color.z - textureData.z); 
    colorOUT.w = 0;
} 

