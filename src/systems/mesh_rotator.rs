//! Simple mesh rotator implementation

use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::core::world::World;
use crate::systems::system::System;
use glam::Mat4;

/// System that will rotate meshes in the scene by a certain amount every frame
pub struct MeshRotatorSystem
{
    /// Rotation in degrees per second
    speed: f32,
}

impl System for MeshRotatorSystem
{
    fn update(&self, delta_time: f32, world: &World)
    {
        let angle_rad = (self.speed * delta_time).to_radians();

        for (_, transform) in world.query_pair::<Mesh, Transform>()
        {
            let mut transform_mut = transform.borrow_mut();

            if !transform_mut.is_static
            {
                transform_mut.matrix *= Mat4::from_rotation_y(angle_rad);
                transform_mut.has_been_changed = true;
            }
        }
    }
}

/// Custom rotation speed default value implementation
impl Default for MeshRotatorSystem
{
    fn default() -> Self
    {
        MeshRotatorSystem { speed: 15.0 }
    }
}
