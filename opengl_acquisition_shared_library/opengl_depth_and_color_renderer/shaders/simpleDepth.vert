#version 130


uniform mat4 MVP;

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
 
out vec3 color;

float packColor(vec3 color) 
{
    return color.r + color.g * 256.0 + color.b * 256.0 * 256.0;
}

vec3 unpackColor(float f) 
{
    vec3 color;
    color.b = floor(f / 256.0 / 256.0);
    color.g = floor((f - color.b * 256.0 * 256.0) / 256.0);
    color.r = floor(f - color.b * 256.0 * 256.0 - color.g * 256.0);
    // now we have a vec3 with the 3 components in range [0..255]. Let's normalize it!
    return color / 255.0;
}

void main()
{
    //Draw like you would normally draw in a shader
    //color = vColor;
 
    //Go For Depth
    gl_Position = MVP *  vec4(vPosition,1.0);
    
    float depth;
    depth = 1000 * gl_Position.z;///gl_Position.w;
    //depth = (-gl_Position.z-gl_DepthRange.near)/(gl_DepthRange.far-gl_DepthRange.near); // will map near..far to 0..1
    //depth = (gl_Position.z-gl_DepthRange.near)/(gl_DepthRange.far-gl_DepthRange.near); // will map near..far to 0..1
    
    color=unpackColor(depth); 

    //color=vec3(depth,depth,depth); 
}
