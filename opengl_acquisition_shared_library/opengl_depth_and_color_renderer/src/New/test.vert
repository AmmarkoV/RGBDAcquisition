#version 150 core
 
in  vec3 vPosition; 
in  vec3 vNormal;    
in  vec2 vTexture;    
//in  vec4 vMaterial; // Specular , Diffuse , Ambient , Shine(?)
 
out vec4 color;
out vec4 theLightPosition;
out vec4 theLightDirection;
out vec4 theNormal;
out vec4 theV;
out vec2 theTexCoords;
 
uniform vec4 fogColorAndScale;

uniform vec3 lightPosition;
uniform vec4 lightColor;
uniform vec4 lightMaterials;
uniform vec4 materialColor;


uniform mat4 modelViewProjection;
uniform mat4 modelView;
uniform mat4 normalTransformation;


void main()
{  
    vec4 vPositionHom = vec4(vPosition,1.0);
     
    theV = modelViewProjection  * vPositionHom;
    theNormal = normalTransformation * vec4(vNormal,1.0);
    
    theLightPosition=vec4(lightPosition,1.0) - vPositionHom; 
    theLightDirection=-1*theLightPosition;
    normalize(theLightPosition);    
    normalize(theLightDirection);    
 
    //gl_Position = vec4(vPosition,1.0);
    gl_Position = theV; 
    theTexCoords = vTexture;
    color = materialColor; 
}
 
 
 
