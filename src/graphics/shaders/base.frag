// Base fragment shader

#version 450

// Input fragment data (coming from vertex shader)
layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv0;
layout(location = 3) in vec2 v_uv1;

// Output fragment data
layout(location = 0) out vec4 f_color;

// Texture samplers
layout(set = 1, binding = 0) uniform sampler2D albedoTex;

// Lighting settings + environment/irradiance map sampler
layout(set = 2, binding = 0) uniform EnvironmentMapData
{
    float exposure;
    float angle;
} lighting_settings;

layout(set = 2, binding = 1) uniform samplerCube irradianceMap;

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

void main()
{
    vec4 albedo = texture(albedoTex, v_uv0);
    vec4 irradiance = texture(irradianceMap, rotateY(v_normal, lighting_settings.angle));

    f_color = albedo * irradiance;
}