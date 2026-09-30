// Tonemapper fragment shader (as a subpass)

#version 450

// Input HDR color
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput s_color;

// Output SDR color
layout(location = 0) out vec4 f_color;

// Post process extra parameters (not used in this shader, but added as reference for future post processes)
layout(push_constant) uniform PostProcessShaderData
{
    float strength;
} parameters;

void main()
{
    // Tonemapping + Gamma correction
    vec3 hdrColor = subpassLoad(s_color).rgb;

    vec3 tonemapped_color = hdrColor / (hdrColor + vec3(1));
    vec3 gamma_corrected_color = pow(tonemapped_color, vec3(1.0/2.2));

    f_color = vec4(gamma_corrected_color, 1);
}