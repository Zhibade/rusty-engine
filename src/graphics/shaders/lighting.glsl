// Include file with lighting data

// ------
// Buffers and samplers
// ------

// Lighting settings + environment/irradiance map sampler
layout(set = 3, binding = 0) uniform EnvironmentMapData
{
    float exposure;
    float angle;
} lighting_settings;

layout(set = 3, binding = 1) uniform samplerCube irradianceMap;
layout(set = 3, binding = 2) uniform samplerCube prefilteredMap;
layout(set = 3, binding = 3) uniform sampler2D brdfIntegrationMap;

// PBR parameters
layout(set = 4, binding = 0) uniform PbrParameters
{
    vec4 color_tint;
    float occlusion_strength;
    float roughness_strength;
    float metallic_strength;
    float normal_strength;
} pbr_parameters;