//! Post process component implementation

use super::component::Component;
use derive_macros::Component;
use std::any::Any;

/// Post process that will be used when rendering the scene
/// (Only HDR color input is supported at the moment)
#[derive(Component)]
pub struct PostProcess
{
    pub shader_path: String, // Fragment shader path
    pub strength: f32,
    pub priority: u32, // Determines rendering order of post processes (0 = highest priority)
    pub was_updated: bool, // Set this to true when a property has been changed

    /// This property is meant to be handled via the systems, do not set it manually to true during
    /// initialization or otherwise
    pub has_been_initialized: bool,
}

/// Default trait implementation
impl Default for PostProcess
{
    fn default() -> Self
    {
        PostProcess {
            shader_path: String::new(),
            strength: 1.0,
            priority: 0,
            was_updated: true,
            has_been_initialized: false
        }
    }
}
