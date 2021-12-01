#version 130

uniform mat4 MVP;

uniform vec3 iResolution;
uniform float iTime;

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
in  vec2 vTexture;   
 
out vec4 color;
out vec2 UV;
 
void main()
{
    vec4 vColor = vec4(vColor,1.0); 
    
    gl_Position = MVP *  vec4(vPosition,1.0);

    color = vColor;
    UV   =  vTexture;
}
