//! Material component implementation

use super::component::Component;
use derive_macros::Component;
use std::any::Any;

pub const DEFAULT_COLOR_TEXTURE_PATH: &str = "default-white";
pub const DEFAULT_NORMAL_TEXTURE_PATH: &str = "default-normal";
pub const DEFAULT_ORM_TEXTURE_PATH: &str = "default-orm";

/// Lighting mode for use in a material
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum LightingMode
{
    Unlit,
    Lit,
    PBR,
}

/// Texture usage in the shader
#[derive(Copy, Clone, Eq, Hash, PartialEq)]
pub enum TextureUsage
{
    Color = 0,
    OcclusionRoughnessMetallic = 1,
    Normal = 2,
}

impl Default for TextureUsage
{
    fn default() -> Self
    {
        TextureUsage::Color
    }
}

/// PBR parameters (only used when [LightingMode] is [PBR](LightingMode::PBR)
pub struct PbrParameters
{
    pub color_tint: [f32; 4],
    pub occlusion_strength: f32,
    pub roughness_strength: f32,
    pub metallic_strength: f32,
    pub normal_strength: f32,

    /// This property is meant to be handled via the systems, do not set it manually to true during
    /// initialization or otherwise
    pub was_updated: bool,
}

impl Default for PbrParameters
{
    fn default() -> Self
    {
        PbrParameters {
            color_tint: [1.0, 1.0, 1.0, 1.0],
            occlusion_strength: 1.0,
            roughness_strength: 1.0,
            metallic_strength: 1.0,
            normal_strength: 1.0,
            was_updated: false,
        }
    }
}

/// Texture parameters for use with a material/shader\
/// NOTE: This is not the image contents, the image contents have to be created with
/// [image_loader]'s [load_image](image_loader::load_image) and then initialized with
/// [GfxApi]'s [init_texture](GfxApi::init_texture) before they can be used in a shader
#[derive(Default)]
pub struct Texture
{
    pub filepath: String,
    pub is_cubemap: bool,
    pub usage: TextureUsage,
}

impl Texture
{
    /// Get default white color texture
    pub fn default_color() -> Texture
    {
        Texture {
            filepath: String::from(DEFAULT_COLOR_TEXTURE_PATH),
            is_cubemap: false,
            usage: TextureUsage::Color,
        }
    }

    /// Get default occlusion-roughness-metallic texture
    pub fn default_normal() -> Texture
    {
        Texture {
            filepath: String::from(DEFAULT_NORMAL_TEXTURE_PATH),
            is_cubemap: false,
            usage: TextureUsage::Normal,
        }
    }

    /// Get default occlusion-roughness-metallic texture
    pub fn default_orm() -> Texture
    {
        Texture {
            filepath: String::from(DEFAULT_ORM_TEXTURE_PATH),
            is_cubemap: false,
            usage: TextureUsage::OcclusionRoughnessMetallic,
        }
    }
}

/// Shader parameters that will be used when rendering a mesh
#[derive(Component)]
pub struct Material
{
    // TODO: Multiple material support for one mesh/entity
    pub name: String,
    pub vertex_shader_path: String,
    pub fragment_shader_path: String,
    pub lighting_mode: LightingMode,
    pub textures: Vec<Texture>,
    pub pbr_parameters: Option<PbrParameters>,

    /// This property is meant to be handled via the systems, do not set it manually to true during
    /// initialization or otherwise
    pub has_been_initialized: bool,
}

impl Material
{
    /// Get default empty material
    pub fn default() -> Material
    {
        Material {
            name: String::from("default"),
            vertex_shader_path: "".to_string(),
            fragment_shader_path: "".to_string(),
            lighting_mode: LightingMode::Unlit,
            textures: Vec::new(),
            pbr_parameters: None,
            has_been_initialized: false,
        }
    }

    /// Get default PBR material
    pub fn default_pbr() -> Material
    {
        let color_tex = Texture::default_color();
        let normal_tex = Texture::default_normal();
        let orm_tex = Texture::default_orm();

        Material {
            name: String::from("default-pbr"),
            vertex_shader_path: "src/graphics/shaders/base_vert.spv".to_string(),
            fragment_shader_path: "src/graphics/shaders/base_frag.spv".to_string(),
            lighting_mode: LightingMode::PBR,
            textures: vec![color_tex, normal_tex, orm_tex],
            pbr_parameters: Some(PbrParameters::default()),
            has_been_initialized: false,
        }
    }
}
