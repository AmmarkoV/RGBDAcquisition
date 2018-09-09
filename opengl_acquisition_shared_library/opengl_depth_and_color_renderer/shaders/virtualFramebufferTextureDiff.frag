#version 130

in vec2 UV; 
out vec3 color;

uniform int tileSizeX = 16;
uniform int tileSizeY = 16;

uniform vec3 iResolution;
uniform sampler2D renderedTexture;  

//This is for unsigned short textures
#extension GL_EXT_gpu_shader4 : enable    //Include support for this extension, which defines usampler2D
uniform usampler2D diffedTexture;

uniform float iTime;









#define STARTING_MODULUS 512

bool reduceHere(int mm, vec2 c) 
{
	vec2 m = floor(mod(c, float(mm)));
    return all(equal(m, vec2(0.0)));
}

vec4 T(sampler2D ch, vec2 uv) 
{
	return clamp(uv, 0.0, 1.0) == uv ? texture(ch, uv) : vec4(0.0);   
}

const ivec4 wi = ivec4(1, 2, 4, 8);

vec4 reduce(sampler2D ch0, sampler2D ch1, int mi, vec2 c) {
    vec4 r = vec4(0.0);
    for (int p = 0; p < 4; p++) {
        // this ought to be done with bitwise ops but they are not supported here
        float m = float(wi[p] * mi/2);
        int n = wi[p] * mi;
        if (reduceHere(n, c)) {
            vec3 i = vec3(m, m, 0.0) / iResolution.xyy;
            vec2 uv = c / iResolution.xy;
            vec2 uv_e  = uv + i.xz;
            vec2 uv_ne = uv + i.xy;
            vec2 uv_n  = uv + i.zy;
            if (p == 0) {
                r[p] = T(ch0, uv).w + T(ch0, uv_e).w + T(ch0, uv_ne).w + T(ch0, uv_n).w;
            } else {
                r[p] = T(ch1, uv)[p-1] + T(ch1, uv_e)[p-1] + T(ch1, uv_ne)[p-1] + T(ch1, uv_n)[p-1];
            }
        } else {
           r[p] = 0.0; 
        }
    }
    return r;
}

 

#define thirdByte 65536.0
#define secondByte 256.0
float packColor(vec3 color) 
{
    vec3 denorm = color * 255.0;
    return denorm.r + (denorm.g * secondByte) + (denorm.b * thirdByte);
}

vec3 unpackColor(float f) 
{   
    vec3 color;
    float remaining=round(f);
    
    color.b = floor(remaining / thirdByte );
    remaining = remaining - color.b * thirdByte; 

    color.g = floor(remaining / secondByte);
    remaining = remaining - color.g * secondByte; 

    color.r = remaining; 
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
 
    
    //Difference calculation as Floats
    float diffTextureDepth = float(texture( diffedTexture, verticalFlip ).x);
    float renderTextureDepth = packColor(renderData); 
    float discrepancy = abs(renderTextureDepth - diffTextureDepth);  
    
    if (renderTextureDepth>0)
     {
	  color.x = discrepancy/5000;
      color.y = 0.0;
      color.z = 0.0;   
     }
     
    //If there is both renderings and observations signal it
    if ( (renderTextureDepth>0) && (diffTextureDepth>0) )
    {  
      if (discrepancy<100)
       { 
         color.y = 1.0;       
       }  
   }   
    
    if ( renderTextureDepth>0 )
    {  
      color.z = 1.0;       

    }   

   //If we want to repack everything .. ( we probably don't ) 
   // color = unpackColor(float(discrepancy));
}
 
