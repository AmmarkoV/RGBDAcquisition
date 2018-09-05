#version 130


uniform mat4 MVP;
uniform mat4 MV;

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
 
out vec3 color;
 


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
    //Draw like you would normally draw in a shader
    //color = vColor;
 
    //Calculate final position in window
    gl_Position = MVP *  vec4(vPosition,1.0);
    
    //But we want to store the real position a.k.a. depth as a `color`
    vec4 position3D =  MV*vec4(vPosition,1.0);

    float depth;
    depth = position3D.z;// / position3D.w;
    //depth = (-gl_Position.z-gl_DepthRange.near)/(gl_DepthRange.far-gl_DepthRange.near); // will map near..far to 0..1
    //depth = (depth+gl_DepthRange.near)/(gl_DepthRange.far-gl_DepthRange.near); // will map near..far to 0..1
    
    //if ( (depth>2000) && (depth<3000)) 
    {
     color=unpackColor(depth);  
    } 
    //else { color = vec3(0); }

    //color=vec3(depth,depth,depth); 
}
