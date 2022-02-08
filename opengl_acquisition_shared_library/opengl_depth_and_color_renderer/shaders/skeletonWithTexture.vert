#version 130

uniform mat4 MVP;

uniform vec3 iResolution;
uniform float iTime;

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
in  vec2 vTexture;

//Skeleton Bones
in uvec3 vBoneIndexIDs;
in vec3  vBoneWeightValues;
mat3     vBoneTransform;
 
out vec4 color;
out vec2 UV;
 
void main()
{
    vec4 vColor = vec4(vColor,1.0); 
 
    //Check doModelTransform call in model_loader_transform
    //vec4 skinnedVertex = vec4(0,0,0,0);
    //skinnedVertex += vBoneTransform[vBoneIndexIDs.w] * vec4(vPosition,1.0) * vBoneWeights.w;
    
    vec3 skinnedVertex = vec3(0,0,0);
    skinnedVertex += vBoneTransform[vBoneIndexIDs.x] * vPosition * vBoneWeightValues.x;
    skinnedVertex += vBoneTransform[vBoneIndexIDs.y] * vPosition * vBoneWeightValues.y;
    skinnedVertex += vBoneTransform[vBoneIndexIDs.z] * vPosition * vBoneWeightValues.z;    

    //Todo: also transform  normal (which is not currently used)
    gl_Position = MVP *  vec4(skinnedVertex.xyz,1.0);

    color = vColor;
    UV   =  vTexture;
}
