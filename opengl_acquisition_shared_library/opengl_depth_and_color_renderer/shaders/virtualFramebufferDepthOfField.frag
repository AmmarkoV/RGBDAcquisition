#version 130

in vec2 UV;

out vec3 color;

uniform vec3 iResolution;
uniform sampler2D renderedTexture; 
uniform float iTime;


const float GA =2.399; 
const mat2 rot = mat2(cos(GA),sin(GA),-sin(GA),cos(GA));

// 	simplyfied version of Dave Hoskins blur
vec3 dof(sampler2D tex,vec2 uv,float rad)
{
	vec3 acc=vec3(0);
    vec2 pixel=vec2(.002*iResolution.y/iResolution.x,.002),angle=vec2(0,rad);;
    rad=1.;
	for (int j=0;j<80;j++)
    {  
        rad += 1./rad;
	    angle*=rot;
        vec4 col=texture(tex,uv+pixel*(rad-1.)*angle);
		acc+=col.xyz;
	}
	return acc/80.;
}


void main()
{  
    //color = texture( renderedTexture, UV  ).xyz;
    
    //Depth Of Field 
    color =  dof(renderedTexture,UV,1.5);
}
