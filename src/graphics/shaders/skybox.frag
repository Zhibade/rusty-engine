// Skybox fragment shader

#version 450

// Input fragment data (coming from vertex shader)
layout(location = 0) in vec3 v_view_dir;

// Output fragment data
layout(location = 0) out vec4 f_color;

// Texture samplers
layout(set = 2, binding = 0) uniform samplerCube skyboxTex;

void main()
{
    vec3 sky = texture(skyboxTex, v_view_dir).rgb;

    // Tonemapping & gamma correction
    vec3 tonemapped_sky = sky / (sky + vec3(1));
    vec3 gamma_corrected_sky = pow(sky, vec3(1.0 / 2.2));

    f_color = vec4(gamma_corrected_sky, 1);
}