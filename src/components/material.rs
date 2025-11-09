//! Material component implementation

use super::component::Component;
use derive_macros::Component;
use std::any::Any;

/// Texture usage in the shader
#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub enum TextureUsage
{
    Albedo = 0,
}

impl Default for TextureUsage
{
    fn default() -> Self
    {
        TextureUsage::Albedo
    }
}

/// Texture parameter for use with a material/shader\
/// NOTE: This is not the image contents, the image contents have to be created with
/// [image_loader]'s [load_image] and then initialized with [GfxApi]'s [init_texture]
/// before they can be used in a shader
#[derive(Default)]
pub struct Texture
{
    pub filepath: String,
    pub is_rgb: bool,
    pub usage: TextureUsage,
}

/// Shader parameters that will be used when rendering a mesh
#[derive(Component, Default)]
pub struct Material
{
    pub has_been_initialized: bool,
    pub textures: Vec<Texture>,
}
