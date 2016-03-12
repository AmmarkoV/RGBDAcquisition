#version 150 core

in float fog;
in vec4 theNormal;
in vec4 theV;
in vec4 theLight;
in vec4 color;

out  vec4  colorOUT;
 

uniform vec4 fogColorAndScale; 
uniform vec4 lightColor;
uniform vec4 lightMaterials;

void DirectionalLight( 
                      in vec4 lightPosition,
                      in vec4 halfVector,
                      in vec4 normal,
                      inout vec4 ambient,
                      inout vec4 diffuse,
                      inout vec4 specular
                      )
{
  float shininess=0.3;
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






void main()
{ 

   vec4 L = theLight;   
   vec4 E = normalize(-theV); // we are in Eye Coordinates, so EyePos is (0,0,0)  
   //vec4 R = normalize(-reflect(L,theNormal));  
   vec4 R = normalize( L + E );
 

    vec4 Ambient=vec4(0.0,0.0,0.0,0.0);  
    vec4 Diffuse=vec4(0.0,0.0,0.0,0.0);   
    vec4 Specular=vec4(0.0,0.0,0.0,0.0);
    

     DirectionalLight( 
                      theLight ,
                      R,
                      theNormal ,
                      Ambient,
                      Diffuse,
                      Specular
                      );


    colorOUT = color + Ambient   + Diffuse + Specular;
    colorOUT[3]=1.0;
        
    //colorOUT = vec4(0.0,1.0,0.0,1.0); 
    //colorOUT = color; //No shading done
       
    //Add Fog
    //colorOUT = mix(fogColorAndScale, colorOUT, 1.0-fog);

    colorOUT = clamp(colorOUT, 0.0, 1.0);    
  
}

