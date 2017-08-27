#version 150 core
 
in  vec3 vPosition; 
in  vec3 vNormal;    
in  vec2 vTexture;    
 
out vec4 color;
 
out vec4 theNormal;
out vec4 theV;
out vec3 theTexCoords;


uniform vec4 fogColorAndScale;
uniform mat4 modelViewProjection;
uniform mat4 modelView;
uniform mat4 normalTransformation;


void main()
{
    vec4 vColor = vec4(1.0,0.0,0.0,1.0);  

    vec4 vPositionHom = vec4(vPosition,1.0);
     
    theV = modelViewProjection  * vPositionHom;
    theNormal = normalTransformation * vec4(vNormal,1.0);
     
  
    gl_Position = theV; 
    theTexCoords = vPosition; // vec3(vTexture,1.0);
    color = vColor; 

}
