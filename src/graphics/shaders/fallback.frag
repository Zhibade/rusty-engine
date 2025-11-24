// Fallback fragment shader

#version 450

// Output fragment data
layout(location = 0) out vec4 f_color;

void main()
{
    f_color = vec4(1, 0, 1, 1);
}