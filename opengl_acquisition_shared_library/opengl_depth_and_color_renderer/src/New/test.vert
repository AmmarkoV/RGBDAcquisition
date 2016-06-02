#version 150 core
 
in  vec3 vPosition; 
in  vec3 vNormal;   
in  vec3 vColor;     
in  vec2 vTexture;    
//in  vec4 vMaterial; // Specular , Diffuse , Ambient , Shine(?)
 
out vec4 color;
out vec4 theLightPosition;
out vec4 theLightDirection;
out vec4 theNormal;
out vec4 theV;
out vec2 theTexCoords;
out vec3 theEnvReflectDirection;
 
uniform vec4 fogColorAndScale;

uniform vec3 lightPosition;
uniform vec4 lightColor;
uniform vec4 lightMaterials;
uniform vec4 materialColor;
 
uniform vec4 HDRSettings;

uniform mat4 modelViewProjection;
uniform mat4 modelView;
uniform mat4 normalTransformation;





const float C1 = 0.429043;
const float C2 = 0.511664;
const float C3 = 0.743125;
const float C4 = 0.886227;
const float C5 = 0.247708;
  
// Constants for Old Town Square lighting
const vec3 L00 = vec3( 0.871297, 0.875222, 0.864470);
const vec3 L1m1 = vec3( 0.175058, 0.245335, 0.312891);
const vec3 L10 = vec3( 0.034675, 0.036107, 0.037362);
const vec3 L11 = vec3(-0.004629, -0.029448, -0.048028);
const vec3 L2m2 = vec3(-0.120535, -0.121160, -0.117507);
const vec3 L2m1 = vec3( 0.003242, 0.003624, 0.007511);
const vec3 L20 = vec3(-0.028667, -0.024926, -0.020998);
const vec3 L21 = vec3(-0.077539, -0.086325, -0.091591);
const vec3 L22 = vec3(-0.161784, -0.191783, -0.219152);

 

void computeSphericalHarmonics( in vec4 theNormal , float scaling , inout vec3 diffuseSphHar )
{
   diffuseSphHar = C1 * L22 * ( theNormal.x * theNormal.x - theNormal.y * theNormal.y ) + 
                   C3 * L20 * theNormal.z * theNormal.z +
                   C4 * L00 - 
                   C5 * L20 +
                   2.0 * C1 * L2m2 * theNormal.x * theNormal.y +  
                   2.0 * C1 * L21  * theNormal.x * theNormal.z +  
                   2.0 * C1 * L2m1 * theNormal.y * theNormal.z +  
                   2.0 * C2 * L11  * theNormal.x +  
                   2.0 * C2 * L1m1 * theNormal.y +  
                   2.0 * C2 * L10  * theNormal.z;   
   diffuseSphHar *= scaling;

}


void main()
{  
    vec4 vPositionHom = vec4(vPosition,1.0);
     
    theV = modelViewProjection  * vPositionHom;
    theNormal = normalTransformation * vec4(vNormal,1.0);
    
    theLightPosition=vec4(lightPosition,1.0) - vPositionHom; 
    theLightDirection=-1*theLightPosition;
    normalize(theLightPosition);    
    normalize(theLightDirection);    
  
    gl_Position = theV; 
    theTexCoords = vTexture;
   
    //vec3 diffuseSphHar = materialColor.xyz;    
    //computeSphericalHarmonics (theNormal,HDRSettings[2],diffuseSphHar);
    //color = vec4(diffuseSphHar,1.0); 

    color = vec4(vColor,1.0);
            
    theEnvReflectDirection= normalize(reflect(theV.xyz,vNormal)); 
}
 
 
 
