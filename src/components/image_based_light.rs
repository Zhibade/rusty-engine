//! Image-based lighting component implementation

use super::component::Component;
use derive_macros::Component;
use std::any::Any;

/// Image-based lighting component\
/// Contains the necessary data for doing image-based environment lighting (angle is in degrees)
#[derive(Component)]
pub struct ImageBasedLight
{
    pub exposure: f32,
    pub angle: f32,
    pub cubemap: String,
    pub already_initialized: bool,
}

impl Default for ImageBasedLight
{
    fn default() -> Self
    {
        ImageBasedLight {
            exposure: 1.0,
            angle: 0.0,
            cubemap: String::from(""),
            already_initialized: false,
        }
    }
}
