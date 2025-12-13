// Skybox fragment shader

#version 450

// Input fragment data (coming from vertex shader)
layout(location = 0) in vec3 v_view_dir;

// Output fragment data
layout(location = 0) out vec4 f_color;

// Texture samplers
layout(set = 1, binding = 0) uniform samplerCube skyboxTex;

void main()
{
    f_color = texture(skyboxTex, v_view_dir);
}