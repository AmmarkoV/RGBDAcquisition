#version 130

in vec2 UV;

out vec3 color;

uniform vec3 iResolution;
uniform sampler2D renderedTexture; 
uniform float iTime;
void main()
{  
    vec2 uvn = UV - vec2(0.5); 
    float d=length(uvn);  
  
    //Normal 
  	//color = texture( renderedTexture, vec2(UV.x, UV.y) ).xyz ;
    
    //Vignette
    color =  texture( renderedTexture, UV  ).xyz - vec3(d) ;
}
 
