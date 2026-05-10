// Base vertex shader

#version 450

// Input vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 uv0;
layout(location = 4) in vec2 uv1;
layout(location = 5) in vec3 color;

// Extra output vertex data
layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 v_uv0;
layout(location = 2) out vec2 v_uv1;
// Tangent + bitangent + normal matrix for normal mapping
// Matrices take several locations, so it is better to have it at the end to avoid having to manually adjust them
layout(location = 3) out mat3 v_tbn_matrix;

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
    // Transform vertex position to match object's transform + camera transform + perspective
    mat4 view_perspective_matrix = view_perspective_data.perspective_matrix * view_perspective_data.view_matrix;
    gl_Position = view_perspective_matrix * object_transform_data.matrix * vec4(position, 1.0);

    // Transform normal and tangent, then calculate bitangent for normal mapping
    // Can't transform directly with object's transform matrix because non-uniform scale would cause normals to be wrong
    // Need to calculate a new matrix for transforming normals that works with non-uniform scale (apply inverse scale
    // to the normal (inverse), but retain the rotations (transpose))
    mat3 normalMatrix = transpose(inverse(mat3(object_transform_data.matrix)));
    vec3 normal = normalize(normalMatrix * normal);
    vec3 tangent = normalize(normalMatrix * tangent);
    vec3 bitangent = cross(normal, tangent);

    v_tbn_matrix = mat3(tangent, bitangent, normal);

    // Passthrough the remaining properties
    v_color = color;
    v_uv0 = uv0;
    v_uv1 = uv1;
}