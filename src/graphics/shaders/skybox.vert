// Skybox vertex shader

#version 450

// Input vertex data
layout(location = 0) in vec3 position;

// Extra output vertex data
layout(location = 0) out vec3 v_view_dir;

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
    v_view_dir = normalize(position);

    mat4 view_perspective_matrix = view_perspective_data.perspective_matrix * view_perspective_data.view_matrix;
    gl_Position = view_perspective_matrix * object_transform_data.matrix * vec4(position, 1.0);
}