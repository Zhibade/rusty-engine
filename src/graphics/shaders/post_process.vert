// Tonemapper vertex shader (hard-coded fullscreen triangle)

#version 450

void main()
{
    // Triangle vertices go outside of the edges of the screen so that a single triangle covers the entire screen
    vec2 outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0f + -1.0f, 0.0f, 1.0f);
}