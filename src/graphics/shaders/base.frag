// Base fragment shader

#version 450

// Input fragment data (coming from vertex shader)
layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_uv0;
layout(location = 2) in vec2 v_uv1;

// Output fragment data
layout(location = 0) out vec4 f_color;

// Texture samplers
layout(set = 1, binding = 0) uniform sampler2D albedoTex;

void main()
{
    f_color = texture(albedoTex, v_uv0);
}