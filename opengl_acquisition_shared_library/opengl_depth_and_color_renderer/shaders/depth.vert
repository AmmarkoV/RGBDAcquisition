#version 130

//https://stackoverflow.com/questions/5669287/opengl-compute-eye-space-coord-from-window-space-coord-in-glsl

uniform mat4 MVP;
uniform mat4 MVPInverse;

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
 
out vec4 color;
 
  
uniform vec2 iResolution; // window width, height
uniform vec2 clipPlanes;  // zNear, zFar 

#define ZNEAR clipPlanes.x
#define ZFAR clipPlanes.y

#define A (ZNEAR + ZFAR)
#define B (ZNEAR - ZFAR)
#define C (2.0 * ZNEAR * ZFAR)
#define D (ndcPos.z * B)
#define ZEYE -(C / (A + D))

void main() 
{
vec3 ndcPos;
ndcPos.xy = vPosition.xy / iResolution;
ndcPos.z =  vPosition.z;// or texture2D (sceneDepth, ndcPos.xy).r; // or gl_FragCoord.z
ndcPos -= 0.5;
ndcPos *= 2.0;
vec4 clipPos;
clipPos.w = -ZEYE;
clipPos.xyz = ndcPos * clipPos.w;
vec4 eyePos = MVPInverse * clipPos;
}
