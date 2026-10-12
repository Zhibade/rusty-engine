//! Transform component implementation

use crate::components::component::Component;
use derive_macros::Component;
use glam::{Mat4, Vec3};
use std::any::Any;

/// Contains an entity's 3D transformation data
#[derive(Component, Default)]
pub struct Transform
{
    pub has_been_changed: bool,
    pub is_static: bool,
    pub matrix: Mat4,
}

impl Transform
{
    /// Create a new [Transform] component using the given position
    pub fn from_position(position: Vec3) -> Self
    {
        Transform {
            matrix: Mat4::from_translation(position),
            has_been_changed: false,
            is_static: false,
        }
    }

    /// Create a new [Transform] component using the given scale
    pub fn from_scale(scale: Vec3) -> Self
    {
        Transform {
            matrix: Mat4::from_scale(scale),
            has_been_changed: false,
            is_static: false
        }
    }
}
