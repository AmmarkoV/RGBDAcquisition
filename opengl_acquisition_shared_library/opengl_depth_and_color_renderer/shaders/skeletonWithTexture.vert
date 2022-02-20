#version 130

uniform mat4 MVP;

uniform vec3 iResolution;
uniform float iTime;

const int MAX_BONES = 171; 		
uniform mat4 vBoneTransform[MAX_BONES];

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
in  vec2 vTexture;

//Skeleton Bones
in uvec3 vBoneIndexIDs;
in vec3  vBoneWeightValues;
 
out vec4 color;
out vec2 UV;
 
void main()
{
    vec4 vColor = vec4(vColor,1.0); 
 
    //Check doModelTransform call in model_loader_transform
    vec4 skinnedVertex = vec4(vPosition,1.0); 
    //vec4 skinnedVertex = vec4(0,0,0,0);
    skinnedVertex += vBoneTransform[vBoneIndexIDs.x] * vec4(vPosition,1.0) * vBoneWeightValues.x;
    skinnedVertex += vBoneTransform[vBoneIndexIDs.y] * vec4(vPosition,1.0) * vBoneWeightValues.y;
    skinnedVertex += vBoneTransform[vBoneIndexIDs.z] * vec4(vPosition,1.0) * vBoneWeightValues.z;    
    //skinnedVertex = normalize(skinnedVertex);   
    //skinnedVertex.w=1; 


    //https://ogldev.org/www/tutorial38/tutorial38.html
    //Todo: also transform  normal (which is not currently used)
    //gl_Position = MVP *  skinnedVertex;
    gl_Position = MVP *  vec4(skinnedVertex.xyz,1.0);

    //This is simple rendering without skinned
    //gl_Position = MVP * vec4(vPosition,1.0);
    //gl_Normal   = MVP * vec4(vPosition,0.0);

    color = vColor;
    UV   =  vTexture;
}
