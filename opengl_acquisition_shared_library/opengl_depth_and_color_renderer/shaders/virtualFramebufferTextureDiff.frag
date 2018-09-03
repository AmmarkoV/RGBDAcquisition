#version 130

in vec2 UV; 
out vec3 color;

uniform int tileSizeX = 16;
uniform int tileSizeY = 16;

uniform vec3 iResolution;
uniform sampler2D renderedTexture; 
//This is for Color Textures
//uniform sampler2D diffedTexture; 

//This is for unsigned short textures
#extension GL_EXT_gpu_shader4 : enable    //Include support for this extension, which defines usampler2D
uniform usampler2D diffedTexture;

uniform float iTime;


float packColor(vec3 color) {
    return color.r + color.g * 256.0 + color.b * 256.0 * 256.0;
}


uint packColorU(vec3 color) 
{
    float intermediate = color.r + color.g * 256.0 + color.b * 256.0 * 256.0;
    return uint(intermediate);
}

vec3 unpackColor(float f) {
    vec3 color;
    color.b = floor(f / 256.0 / 256.0);
    color.g = floor((f - color.b * 256.0 * 256.0) / 256.0);
    color.r = floor(f - color.b * 256.0 * 256.0 - color.g * 256.0);
    // now we have a vec3 with the 3 components in range [0..255]. Let's normalize it!
    return color / 255.0;
}


void main()
{ 
    vec3 renderData = texture( renderedTexture , UV ).xyz ; 
    
    //Get correct texel for our diff/tiled texure
    vec2 UVDiffTexture;
    UVDiffTexture.x = UV.x*tileSizeX;
    UVDiffTexture.y = UV.y*tileSizeY;
    vec2 verticalFlip=UVDiffTexture;
    verticalFlip.y = 1.0f - verticalFlip.y;

    //Difference calculation as Uints
    //uint diffTextureDepth = texture( diffedTexture, verticalFlip ).x;
    //uint renderTextureDepth = packColorU(renderData); 
    //uint discrepancy = abs(renderTextureDepth - diffTextureDepth); 
    
    //Difference calculation as Floats
    float diffTextureDepth = texture( diffedTexture, verticalFlip ).x;
    float renderTextureDepth = packColor(renderData); 
    float discrepancy = abs(renderTextureDepth - diffTextureDepth); 
    
    //if ( int(renderTextureDepth)!=0)
     {
	  color.x = discrepancy;
      color.y = discrepancy; 
      color.z = discrepancy;  
     }
     
    if ( (int(renderTextureDepth)>0) && (int(diffTextureDepth)>0) )
    { 
      color.y = 1.0; 
    } 
    
    color = unpackColor(float(discrepancy));

    //color = diffData;
	//color = texture( renderedTexture, UV ).xyz ;
	//color = texture( renderedTexture, UV + 0.005*vec2( sin(iTime+1024.0*UV.x),cos(iTime+768.0*UV.y)) ).xyz ; 
}
 
