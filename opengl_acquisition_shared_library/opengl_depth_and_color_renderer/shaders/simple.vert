#version 130


uniform mat4 MVP;

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
 
out vec4 color;
 

void main()
{
    vec4 vColor = vec4(vColor,1.0); 
    gl_Position = MVP *  vec4(vPosition,1.0);
    color = vColor;
}
