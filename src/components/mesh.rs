///! Rendereable 3D mesh component
use glam::{Vec2, Vec3, Vec4};

// Represents a 3D mesh for rendering
// Contains the necessary data for rendering a mesh such as vertices, normals, etc
#[derive(Default)]
pub struct Mesh
{
    pub vertices: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub normals: Vec<Vec3>,
    pub tangents: Vec<Vec3>,
    pub colors: Vec<Vec4>,
    pub uv_sets: Vec<Vec<Vec2>>,
}
