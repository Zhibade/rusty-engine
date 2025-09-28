// Placeholder vertex shader

#version 450

// Input vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

// Extra output vertex data
layout(location = 0) out vec3 v_color;

// View and perspective projection matrices as a uniform buffer
layout(binding = 0) uniform ViewPerspectiveData
{
    mat4 view_matrix;
    mat4 perspective_matrix;
} view_perspective_data;

// Object's transform as a push constant
layout(push_constant) uniform ObjectTransformData
{
    mat4 matrix;
} object_transform_data;

void main()
{
    v_color = color;

    mat4 view_perspective_matrix = view_perspective_data.perspective_matrix * view_perspective_data.view_matrix;
    gl_Position = view_perspective_matrix * object_transform_data.matrix * vec4(position, 1.0);
}