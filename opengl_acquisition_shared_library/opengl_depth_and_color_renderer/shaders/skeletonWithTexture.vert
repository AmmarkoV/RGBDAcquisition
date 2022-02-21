#version 130

uniform mat4 MVP;

uniform vec3 iResolution;
uniform float iTime;

const int MAX_BONES = 200; //Max used is 172, GPU max accomodation is 1000		
uniform mat4 vBoneTransform[MAX_BONES];

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
in  vec2 vTexture;

//Skeleton Bones
in uvec4 vBoneIndexIDs;
in vec4  vBoneWeightValues;
 
out vec4 color;
out vec2 UV;
 
void main()
{
    vec4 vColor = vec4(vColor,1.0); 

    //Check doModelTransform call in model_loader_transform for CPU way to do this..
    vec4 skinnedVertex  = vBoneTransform[vBoneIndexIDs[0]] * (vBoneWeightValues[0] * vec4(vPosition,1.0));
         skinnedVertex += vBoneTransform[vBoneIndexIDs[1]] * (vBoneWeightValues[1] * vec4(vPosition,1.0));
         skinnedVertex += vBoneTransform[vBoneIndexIDs[2]] * (vBoneWeightValues[2] * vec4(vPosition,1.0));
         skinnedVertex += vBoneTransform[vBoneIndexIDs[3]] * (vBoneWeightValues[3] * vec4(vPosition,1.0));        
         skinnedVertex  = normalize(skinnedVertex); 

    //https://ogldev.org/www/tutorial38/tutorial38.html
    //Todo: also transform  normal (which is not currently used)
    gl_Position   = MVP * skinnedVertex;
    //gl_Normal   = MVP * vec4(skinnedVertex.xyz,0.0);
    //gl_Position = MVP * vec4(skinnedVertex.xyz,1.0);


    //This is simple rendering without skinned
    //gl_Position = MVP * vec4(vPosition,1.0);
    //gl_Normal   = MVP * vec4(vPosition,0.0);

    color = vColor;
    UV   =  vTexture;
}
