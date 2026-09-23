// Fallback post process fragment shader

#version 450

// Input HDR color
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput s_color;

// Output SDR color
layout(location = 0) out vec4 f_color;

void main()
{
    f_color = vec4(1, 0, 1, 1);
}