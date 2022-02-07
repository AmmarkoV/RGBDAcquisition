#version 130

uniform mat4 MVP;

uniform vec3 iResolution;
uniform float iTime;

in  vec3 vPosition;
in  vec3 vColor;   
in  vec3 vNormal;   
in  vec2 vTexture;

//Skeleton Bones
in uvec4 bone_ids;
in vec4 bone_weights;
mat4 bones;
 
out vec4 color;
out vec2 UV;
 
void main()
{
    vec4 vColor = vec4(vColor,1.0); 
 
    //Check doModelTransform call in model_loader_transform
    vec4 skinned_vertex = vec4(0,0,0,0);
    skinned_vertex += bones[bone_ids.w] * vec4(vPosition,1.0) * bone_weights.w;
    skinned_vertex += bones[bone_ids.x] * vec4(vPosition,1.0) * bone_weights.x;
    skinned_vertex += bones[bone_ids.y] * vec4(vPosition,1.0) * bone_weights.y;
    //skinned_vertex += bones[bone_ids.z] * vec4(vPosition,1.0) * bone_weights.z;    

    //Todo: also transform  normal (which is not currently used)
    gl_Position = MVP *  vec4(skinned_vertex.xyz,1.0);

    color = vColor;
    UV   =  vTexture;
}
