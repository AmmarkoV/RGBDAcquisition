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

void main()
{ 

   vec4 L = theLight;   
   vec4 E = normalize(-theV); // we are in Eye Coordinates, so EyePos is (0,0,0)  
   //vec4 R = normalize(-reflect(L,theNormal));  
   vec4 R = normalize( L + E );

/*
   vec4 ambient = AmbientProduct;
   float Kd = max(dot(L, N), 0.0);
   vec4 diffuse = Kd*DiffuseProduct;
   float Ks = pow(max(dot(N, H), 0.0), Shininess);
   vec4 specular = Ks*SpecularProduct;
  */ 


    vec4 Ambient = lightColor* lightMaterials[0];

    vec4 Diffuse  = lightColor * pow(max(dot(R,E),0.0),0.3) * lightMaterials[1]; 
    Diffuse = clamp(Diffuse, 0.0, 1.0);     
    
    vec4 Specular = lightColor * pow(max(dot(R,E),0.0),0.3) * lightMaterials[2];   
    Specular = clamp(Specular, 0.0, 1.0);    
    // discard the specular highlight if the light's behind the vertex
    if( dot(L, R) < 0.0 )
      Specular = vec4(0.0, 0.0, 0.0, 1.0);



    colorOUT = color + Ambient   + Diffuse + Specular;
    colorOUT[3]=1.0;
    
    //colorOUT = vec4(0.0,1.0,0.0,1.0); 
    //colorOUT = color; //No shading done
       
    //Add Fog
    //colorOUT = mix(fogColorAndScale, colorOUT, 1.0-fog);
}

