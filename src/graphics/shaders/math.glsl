// Include file with math functions

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