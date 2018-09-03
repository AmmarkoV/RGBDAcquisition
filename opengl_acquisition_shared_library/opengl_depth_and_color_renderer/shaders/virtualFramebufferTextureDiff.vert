#version 130

// Input vertex data, different for all executions of this shader.
in vec3 vertexPosition_modelspace;

// Output data ; will be interpolated for each fragment.
out vec2 UV;
out vec2 UVDiffTexture;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vertexPosition_modelspace.xy+vec2(1,1))/2.0;
    

    UVDiffTexture.x = UV.x*16;
    UVDiffTexture.y = UV.y*16;
}

