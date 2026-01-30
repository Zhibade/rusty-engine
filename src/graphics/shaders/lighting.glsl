// Include file with lighting data and function declarations

// ------
// Constants
// ------

#define PI 3.1415926535897932384626433832795

// ------
// Buffers and samplers
// ------

// Lighting settings + environment/irradiance map sampler
layout(set = 2, binding = 0) uniform EnvironmentMapData
{
    float exposure;
    float angle;
} lighting_settings;

layout(set = 2, binding = 1) uniform samplerCube irradianceMap;

// PBR parameters
layout(set = 3, binding = 0) uniform PbrParameters
{
    vec4 color_tint;
    float occlusion_strength;
    float roughness_strength;
    float metallic_strength;
    float normal_strength;
} pbr_parameters;

// ------
// Functions
// ------

// Rotate direction vector in Y axis
vec3 rotateY(vec3 vector, float angle)
{
    float angleRad = radians(angle);
    mat4 rotMatrix = mat4(cos(angleRad), 0, sin(angleRad), 0,
    0, 1, 0, 0,
    -sin(angleRad), 0, cos(angleRad), 0,
    0, 0, 0, 1);

    return (rotMatrix * vec4(vector, 0)).xyz;
}