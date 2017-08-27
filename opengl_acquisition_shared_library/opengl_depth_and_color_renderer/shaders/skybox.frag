#version 150 core
 
in vec4 theNormal;
in vec4 theV;
in vec3 theTexCoords; 
out  vec4  colorOUT;
 
uniform samplerCube skybox;
uniform vec4 HDRSettings;
uniform vec4 fogColorAndScale; 

void main() 
{   
    vec4 envColor3 = texture(skybox,theTexCoords);   
    float exposure = HDRSettings[0];
    float gamma = HDRSettings[1];
    vec4 mappedEnvColor = vec4(1.0) - exp(-envColor3 * exposure); 
    mappedEnvColor = pow(mappedEnvColor, vec4(1.0 / gamma));
    colorOUT = mappedEnvColor;     


    //Add Fog
    //GL_EXP fog    
    float fogScale = fogColorAndScale.w/10;
    float fog=exp(- fogScale * abs(theV.w));
    fog = clamp(fog, 0.0, 1.0);
    vec4 fogColor = vec4(fogColorAndScale.r,fogColorAndScale.g,fogColorAndScale.b,1.0);     
    colorOUT = mix(fogColor, colorOUT, fog);


} 

