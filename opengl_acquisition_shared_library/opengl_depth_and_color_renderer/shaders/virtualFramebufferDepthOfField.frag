#version 130

in vec2 UV;

out vec3 color;

uniform vec3 iResolution;
uniform sampler2D renderedTexture; 
uniform float iTime;


// Mellow purple flow background - Del 16/06/2018
//https://www.shadertoy.com/view/ldVfWm
const float pi = 3.14159;

vec3 MellowPurple(vec2 pos)
{
    pos *= 0.5;
    vec3 col = vec3(0.32,0.29,0.65);
    float d = length(pos);
    float tt = iTime*0.08;
    float iter = (pos.x*pos.y);
    iter *= 0.5+sin(pos.x*0.5+tt*0.95)+0.5;
    float r = sin(sin(tt)*pos.y);
    r += .4;
    float d1 = sin(tt+pos.x+pos.y);
    float val = sin(tt+iter*pi );
    float brightness = 0.25 / abs( d1 * val - r);
//    brightness +=  0.25 /abs( d1+0.1 * val - r);
//    brightness += 0.25 / abs( d1-0.1 * val - r);
    brightness +=  0.25 /abs( d1+0.4 * val - r);
    brightness += 0.25 / abs( d1-0.4 * val - r);
    brightness *= 0.05;
    brightness = brightness/(brightness + 1.);
    brightness*=0.75-d;
    col += brightness;
    return col;
}


vec3 overlayForeground(vec3 background , vec3 foreground)
{
  if ( dot(foreground,foreground) != 0.0 ) { return foreground; }  
  return background;
}


void main()
{  
	vec2 pos = ( UV - .5 /** iResolution.xy*/ );// / iResolution.y;     
    
    //Just Burn Both
    //color = MellowPurple(pos) + texture( renderedTexture, UV  ).xyz;
    
    //Only Mellow 
    //color = MellowPurple(pos);
      
    //Blend Both
    color = overlayForeground ( MellowPurple(pos) , texture( renderedTexture, UV  ).xyz );
}
