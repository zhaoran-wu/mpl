#version 330 core
out vec4 FragColor;
uniform vec2 zn_zf;

float near = zn_zf.x;//0.1; 
float far  = zn_zf.y;//100.0; 

float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));    
}

void main()
{             
    float depth = LinearizeDepth(gl_FragCoord.z);
    FragColor = vec4(vec3(depth), 1.0);
}