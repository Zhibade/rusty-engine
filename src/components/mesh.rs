//! Rendereable 3D mesh component

use super::component::Component;
use derive_macros::Component;
use glam::{Vec2, Vec3, Vec4};
use std::any::Any;

/// Represents a 3D mesh for rendering
///
/// Contains the necessary data for rendering a mesh such as vertices, normals, etc
#[derive(Component, Default)]
pub struct Mesh
{
    // TODO: Multiple material support for one mesh/entity
    pub filepath: String,
    pub has_been_initialized: bool, // Indicates if mesh data has been uploaded to the GPU or not

    pub vertices: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub normals: Vec<Vec3>,
    pub tangents: Vec<Vec3>,
    pub colors: Vec<Vec4>,
    pub uv_sets: Vec<Vec<Vec2>>,
}
