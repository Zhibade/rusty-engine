// Base fragment shader

#version 450
#include "lighting.glsl"
#include "math.glsl"

// Input fragment data (coming from vertex shader)
layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_uv0;
layout(location = 2) in vec2 v_uv1;
layout(location = 3) in vec3 v_world_pos;
// Tangent + bitangent + normal matrix for normal mapping
// Matrices take several locations, so it is better to have it at the end to avoid having to manually adjust them
layout(location = 4) in mat3 v_tbn_matrix;

// Output fragment data
layout(location = 0) out vec4 f_color;

// Shared extra data
layout(set = 1, binding = 0) uniform FragmentShaderExtraDataContents
{
    vec3 camera_pos;
} extra_data;

// Texture samplers
layout(set = 2, binding = 0) uniform sampler2D albedoTex;
layout(set = 2, binding = 1) uniform sampler2D ormTex; // Occlusion-Roughness-Metallic
layout(set = 2, binding = 2) uniform sampler2D normalTex;

// Constants
const float DIELECTRIC_REFLECTIVITY = 0.04;

void main()
{
    // Texture fetches
    vec3 albedo = pow(texture(albedoTex, v_uv0).rgb, vec3(2.2)); // Gamma -> Linear
    vec4 orm = texture(ormTex, v_uv0);
    vec3 tangentNormal = texture(normalTex, v_uv0).rgb;

    // Final parameter calculations
    albedo *= pbr_parameters.color_tint.rgb;
    float occlusion = orm.r * pbr_parameters.occlusion_strength;
    float roughness = orm.g * pbr_parameters.roughness_strength;
    float metalness = orm.b * pbr_parameters.metallic_strength;
    tangentNormal.rg *= pbr_parameters.normal_strength;
    float exposure = pow(2, lighting_settings.exposure);

    // Calculate world normal from tangent-space normal map
    tangentNormal = tangentNormal * 2 - 1; // Transform from 0-1 to -1-1
    vec3 normal = normalize(v_tbn_matrix * tangentNormal);

    // Calculate reflectivity (F0) based on metalness
    // A constant is used for dielectric materials as an approximation so that we don't have to set it per material type
    vec3 reflectivity = mix(vec3(DIELECTRIC_REFLECTIVITY), albedo.rgb, orm.b);

    // --------------------------
    // -- Image-based lighting --
    // --------------------------
    vec3 view_dir = normalize(extra_data.camera_pos - v_world_pos);
    float NdotV = max(dot(normal, view_dir), 0);

    // specularContribution and diffuseContribution are also commonly known as kS and kD
    vec3 specularContribution = fresnelSchlick(NdotV, reflectivity, roughness);
    vec3 diffuseContribution = (1 - specularContribution) * (1 - metalness); // Metals do not have diffuse reflection

    // Irradiance / Diffuse
    vec3 irradiance = texture(irradianceMap, rotateY(normal, lighting_settings.angle)).rgb;
    vec3 ibl_albedo = exposure * irradiance * albedo * diffuseContribution;

    // Specular/reflections
    vec3 reflectionVector = reflect(-view_dir, normal);

    int prefilteredMapMipmapCount = textureQueryLevels(prefilteredMap);
    vec3 reflection = textureLod(prefilteredMap, rotateY(reflectionVector, lighting_settings.angle), roughness * prefilteredMapMipmapCount).rgb;
    vec2 brdf = texture(brdfIntegrationMap, vec2(NdotV, roughness)).rg;

    vec3 ibl_specular = exposure * reflection * (specularContribution * brdf.x + brdf.y);

    vec3 ibl = (ibl_albedo + ibl_specular) * vec3(occlusion);
    f_color = vec4(ibl, 1); // HDR color is outputted
}