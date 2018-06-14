#version 130


in  vec3 vPosition;
in  vec3 vNormal;   
 
out vec4 color;

uniform vec3 lightPosition;

void main()
{
    vec4 vColor = vec4(1.0,0.0,0.0,1.0); 
    gl_Position = vec4(vPosition,1.0);
    color = vColor;
}
