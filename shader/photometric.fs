#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
// texture samplers
uniform sampler2D texture1;

void main()
{
	FragColor =texture(texture1, TexCoord);
	float expousre_t = 2.3; // 1
	float gamma = 1.167f; //1/1.5
	FragColor.r = FragColor.r*expousre_t ;
	FragColor.g = FragColor.g*expousre_t ;
	FragColor.b = FragColor.b*expousre_t;
  	FragColor.rgb = pow(FragColor.rgb, vec3(gamma));
}