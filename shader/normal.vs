#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
//layout (location = 2) in vec2 aTexCoord;
out vec4 normal_color;

uniform mat4 view;
uniform mat3 normal_R; // normal_R is Rcw 
uniform mat4 projection;


void main()
{
	gl_Position = projection*view*vec4(aPos, 1.0);
	//normal_color = view*vec4(aNormal,1.0);
    vec3 normal_r = normal_R* aNormal; 
    //! todo normal in view? or in pose coodrinate 
    normal_color = vec4(normal_r,1.0);
}