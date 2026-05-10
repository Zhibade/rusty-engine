// Include file with math functions

// ------
// Constants
// ------

#define PI 3.1415926535897932384626433832795

// Get direction vector from sphere coordinates (radius is assumed to be 1)
// Up vector is Y+, so Y and Z axes are swapped from standard equation
vec3 getDirectionVectorFromSphereCoords(float theta, float phi)
{
    return vec3(cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi));
}

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

// Van Der Corput sequence generation using radical inverse method (mirrors a decimal around its decimal point using
// its binary representation)
float vanDerCorputSequence(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

// Generate a sample using a Hammersley sequence method (quasi Monte Carlo sampling)
vec2 hammersleySequence(uint i, uint sampleCount)
{
    return vec2(float(i)/float(sampleCount), vanDerCorputSequence(i));
}

// GGX normal distribution function
// Generates a sample vector based on the given sequence value, point's normal and roughness value
// (Theta <-> Phi angles, Y <-> Z axes are swapped to match renderer's coordinate system)
vec3 ggxSample(vec2 sequenceValue, vec3 normal, float roughness)
{
    float roughnessSqr = roughness * roughness;

    float theta = 2.0 * PI * sequenceValue.x;
    float cosPhi = sqrt((1.0 - sequenceValue.y) / (1.0 + (roughnessSqr * roughnessSqr - 1.0) * sequenceValue.y));
    float sinPhi = sqrt(1.0 - cosPhi * cosPhi);

    // Spherical coordinates to cartesian coordinates
    vec3 coords;
    coords.x = cos(theta) * sinPhi;
    coords.y = cosPhi;
    coords.z = sin(theta) * sinPhi;

    // Tangent-space vector to world-space sample vector
    vec3 up        = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    vec3 sampleVec = tangent * coords.x + normal * coords.y + bitangent * coords.z;
    return normalize(sampleVec);
}

// Slick-GGX geometry self-shadowing approximation function for image-based lighting
float geometrySchlickGGX(float NdotV, float roughness)
{
    float k = (roughness * roughness) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// Smith's geometry self-masking function
float geometrySmith(vec3 normal, vec3 view, vec3 light, float roughness)
{
    float NdotV = max(dot(normal, view), 0.0);
    float NdotL = max(dot(normal, light), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// Schlick's fresnel approximation
// angleDot is the dot product of the view-normal or half vector-view angle
// reflectivity (also known as F0) is the reflection color based on whether it is dielectric or not)
vec3 fresnelSchlick(float angleDot, vec3 reflectivity, float roughness)
{
    float angle = clamp(1 - angleDot, 0, 1);
    vec3 reflection = max(vec3(1.0 - roughness), reflectivity) - reflectivity;

    return reflectivity + reflection * pow(angle, 5.0);
}