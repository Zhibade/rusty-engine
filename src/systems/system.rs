//! System trait implementation

use crate::world::World;

/// Identifies a struct as a system
///
/// This needs to be implemented so that it can be registered and run by a [World]\
/// All structs that implement this trait need to implement the `update` method
pub trait System
{
    /// Will be run every frame by the [World] that it is registered to
    fn update(&self, delta_time: f32, world: &World);
}
