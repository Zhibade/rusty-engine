// Base fragment shader

#version 450
#include "lighting.glsl"

// Input fragment data (coming from vertex shader)
layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv0;
layout(location = 3) in vec2 v_uv1;

// Output fragment data
layout(location = 0) out vec4 f_color;

// Texture samplers
layout(set = 1, binding = 0) uniform sampler2D albedoTex;
layout(set = 1, binding = 1) uniform sampler2D ormTex; // Occlusion-Roughness-Metallic
layout(set = 1, binding = 2) uniform sampler2D normalTex;

void main()
{
    vec4 albedo = texture(albedoTex, v_uv0);
    vec4 irradiance = texture(irradianceMap, rotateY(v_normal, lighting_settings.angle));

    f_color = albedo * irradiance;
}