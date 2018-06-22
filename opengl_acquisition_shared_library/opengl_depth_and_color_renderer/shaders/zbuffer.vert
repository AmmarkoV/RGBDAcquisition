#version 130


uniform mat4 MVP;

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
 
out vec4 color;
 

void main()
{
	//gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    gl_Position = MVP *  vec4(vPosition,1.0);
	float v_depth = gl_Position.z/gl_Position.w; // maybe use: (1.0 + gl_Position.z/gl_Position.w) / 2.0;

    vec4 vColor = vec4(vec3(v_depth),1.0); 
    color = vColor;
}
