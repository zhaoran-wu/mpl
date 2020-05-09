#version 330 core
out vec4 FragColor;
in vec4 normal_color ;

void main()
{
	FragColor =normal_color;
}