// Include file with lighting data

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