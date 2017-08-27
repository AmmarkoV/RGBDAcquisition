#version 150 core 
uniform sampler2D tex1;
uniform sampler2D normalTexture;
uniform samplerCube skybox;

in vec4 theNormal;
in vec4 theV;
in vec4 theLightPosition;
in vec4 theLightDirection;
in vec4 color;
in vec2 theTexCoords;
in vec3 theEnvReflectDirection;



out  vec4  colorOUT; 

uniform vec4 textureStrength;
uniform vec4 fogColorAndScale; 
uniform vec4 lightColor;
uniform vec4 HDRSettings;
uniform vec4 lightMaterials;


 
float shininess=0.3;
float constantAttenuation=0.2;
float linearAttenuation=0.3;
float quadraticAttenuation=0.3;
float spotCosCutoff=0.4;
float spotExponent=0.5;

vec4 fogColor(
          in vec4  rgb,      // original color of the pixel
          float distance, // camera to point distance
          in vec4  rayOri,   // camera position
          in vec4  rayDir 
         )  // camera to point vector
{
  // float fogAmount = c*exp(-rayOri.y*b)*(1.0-exp(-distance*rayDir.y*b))/rayDir.y;
  // vec3  fogColor  = vec3(0.5,0.6,0.7);
  // return mix( rgb, fogColor, fogAmount );
  return rgb;
}


void DirectionalLight( 
                      in vec4 lightPosition,
                      in vec4 halfVector,
                      in vec4 normal,
                      inout vec4 ambient,
                      inout vec4 diffuse,
                      inout vec4 specular
                      )
{
  float nDotVP; // normal . light direction
  float nDotHV; // normal . light half vector
  float pf;     // power factor
  nDotVP = max(0.0, dot(normal,normalize(lightPosition)));
  nDotHV = max(0.0, dot(normal, vec4(halfVector)));
 
  if (nDotVP == 0.0) pf = 0.0; else
                     pf = pow(nDotHV, shininess);

  ambient  += lightColor * lightMaterials[0];
  diffuse  += lightColor * lightMaterials[1] * nDotVP;
  specular += lightColor * lightMaterials[2] * pf;
}

void PointLight( in vec4 lightPosition,
                 in vec4 halfVector,
                 in vec4 normal,
                 in vec4 eye,
                 in vec4 ecPosition3,
                 inout vec4 ambient,
                 inout vec4 diffuse,
                 inout vec4 specular
               )
{
float nDotVP;      // normal . light direction
float nDotHV;      // normal . light half vector
float pf;          // power factor
float attenuation; // computed attenuation factor
float d;           // distance from surface to light source
vec4 VP;           // direction from surface to light position 

// Compute vector from surface to light position
VP = lightPosition - ecPosition3;
// Compute distance between surface and light position
d = length(VP);
// Normalize the vector from surface to light position
VP = normalize(VP);
// Compute attenuation
attenuation = 1.0 / ( constantAttenuation + linearAttenuation * d + quadraticAttenuation * d * d);
//halfVector = normalize(VP + eye);
nDotVP = max(0.0, dot(normal, VP));
nDotHV = max(0.0, dot(normal, halfVector));
if (nDotVP == 0.0) pf = 0.0; else
                   pf = pow(nDotHV, shininess);
ambient  += lightColor * lightMaterials[0] * attenuation;
diffuse  += lightColor * lightMaterials[1] * nDotVP * attenuation;
specular += lightColor * lightMaterials[2] * pf * attenuation;
}

void SpotLight(
               in vec4 lightPosition,
               in vec4 lightDirection,
               in vec4 halfVector,
               in vec4 normal,
               in vec4 eye,
               in vec4 ecPosition3, 
               inout vec4 ambient,
               inout vec4 diffuse,
               inout vec4 specular
              )
{
float nDotVP;  // normal . light direction
float nDotHV;  // normal . light half vector
float pf;      // power factor
float spotDot; // cosine of angle between spotlight
float spotAttenuation; // spotlight attenuation factor
float attenuation;     // computed attenuation factor
float d;               // distance from surface to light source
vec4 VP;               // direction from surface to light positionvec3 halfVector;
// direction of maximum highlights
// Compute vector from surface to light position
VP = lightPosition - ecPosition3;
// Compute distance between surface and light position
d = length(VP);
// Normalize the vector from surface to light position
VP = normalize(VP);
// Compute attenuation
attenuation = 1.0 / (constantAttenuation + linearAttenuation * d + quadraticAttenuation * d * d);
// See if point on surface is inside cone of illumination
spotDot = dot(-VP, normalize(lightDirection));
if (spotDot < spotCosCutoff) spotAttenuation = 0.0; else // light adds no contribution
                             spotAttenuation = pow(spotDot, spotExponent);
// Combine the spotlight and distance attenuation.
attenuation *= spotAttenuation;
halfVector = normalize(VP + eye);
nDotVP = max(0.0, dot(normal, VP));
nDotHV = max(0.0, dot(normal, halfVector));
if (nDotVP == 0.0) pf = 0.0; else
                   pf = pow(nDotHV, shininess);
ambient +=  lightColor * lightMaterials[0] * attenuation;
diffuse +=   lightColor * lightMaterials[1] * attenuation;
specular +=  lightColor * lightMaterials[2] * pf * attenuation;
}


void main()
{   
   vec3 texNormal = texture2D(normalTexture, theTexCoords).rgb ;    
   vec4 usedNormal = mix(theNormal,vec4( normalize(texNormal) , 1.0 ),textureStrength[2]); 
     
   vec4 eye=vec4(0.0,0.0,0.0,1.0);  
   vec4 L = theLightPosition;   
   vec4 E = normalize(-theV); // we are in Eye Coordinates, so EyePos is (0,0,0)  
   //vec4 R = normalize(-reflect(L,usedNormal));  
   vec4 R = normalize( L + E );
 

    vec4 Ambient=vec4(0.0,0.0,0.0,0.0);  
    vec4 Diffuse=vec4(0.0,0.0,0.0,0.0);   
    vec4 Specular=vec4(0.0,0.0,0.0,0.0);
    
   if (lightMaterials.w==0.0f)
     { 
      DirectionalLight( 
                      theLightPosition ,
                      R,
                      usedNormal ,
                      Ambient,
                      Diffuse,
                      Specular
                      );
     } else
   if (lightMaterials.w==1.0f) 
     {
      PointLight( theLightPosition,
                 R,
                 usedNormal,
                 eye,
                 eye,
                 Ambient,
                 Diffuse,
                 Specular
               );
     }else
   if (lightMaterials.w==2.0f) 
     {
       SpotLight( theLightPosition,
                  theLightDirection,
                  R,
                  usedNormal,
                  eye,
                  eye,
                  Ambient,
                  Diffuse,
                  Specular
                );
     }
    
    vec4 colorBasic = color;
    vec4 colorTexture = texture2D(tex1,theTexCoords); 
    vec3 envColor3 = vec3(texture(skybox,theEnvReflectDirection));
  
 
    vec4 envColor = vec4(envColor3,1.0);    

    vec4 colorUsed = mix(colorBasic,colorTexture,textureStrength[0]);
    colorUsed = mix(colorUsed,envColor,textureStrength[1]); 





    colorOUT = colorUsed + Ambient   + Diffuse + Specular;
    colorOUT[3]=1.0;
        
    //colorOUT = vec4(0.0,1.0,0.0,1.0); 
    //colorOUT = color; //No shading done 
    //colorOUT = colorBasic; // Just the basic color

    //Add Fog
    //GL_EXP fog    
    float fogScale = fogColorAndScale.w;
    float fog=exp(- fogScale * abs(theV.w));
    fog = clamp(fog, 0.0, 1.0);
    vec4 fogColor = vec4(fogColorAndScale.r,fogColorAndScale.g,fogColorAndScale.b,1.0);     
    colorOUT = mix(fogColor, colorOUT, fog);



    float exposure = HDRSettings[0];
    float gamma = HDRSettings[1];
    vec4 mappedEnvColor = vec4(1.0) - exp(-colorOUT * exposure); 
    mappedEnvColor = pow(mappedEnvColor, vec4(1.0 / gamma));


    colorOUT = clamp(mappedEnvColor, 0.0, 1.0);    
  
}


