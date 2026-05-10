//! Material rendering system implementation

use crate::components::material::{
    DEFAULT_COLOR_TEXTURE_PATH, DEFAULT_NORMAL_TEXTURE_PATH, DEFAULT_ORM_TEXTURE_PATH,
    LightingMode, Material, PbrParameters, Texture, TextureUsage,
};
use crate::core::world::World;
use crate::graphics::gfx_api::GfxApi;
use crate::loaders::image_loader::{ImageData, load_image};
use crate::systems::system::System;
use std::cell::RefCell;
use std::rc::Rc;

/// Material rendering system
///
/// Responsible for initializing and updating shader data
pub struct MaterialRenderingSystem
{
    gfx_api: Rc<RefCell<GfxApi>>,
}

impl MaterialRenderingSystem
{
    /// Creates a new [MaterialRenderingSystem] and passes the given graphics API
    pub fn new(gfx_api: Rc<RefCell<GfxApi>>) -> Self
    {
        MaterialRenderingSystem { gfx_api }
    }

    /// Add any missing PBR [Texture]s (Albedo, ORM, Normal) by using default textures
    fn add_missing_pbr_textures(textures: &mut Vec<Texture>)
    {
        // Albedo texture
        if textures.iter().find(|t| t.usage == TextureUsage::Color).is_none()
        {
            textures.push(Texture::default_color());
        }

        // ORM texture
        if textures.iter().find(|t| t.usage == TextureUsage::OcclusionRoughnessMetallic).is_none()
        {
            textures.push(Texture::default_orm());
        }

        // Normal texture
        if textures.iter().find(|t| t.usage == TextureUsage::Normal).is_none()
        {
            textures.push(Texture::default_normal());
        }
    }
}

impl System for MaterialRenderingSystem
{
    fn update(&self, _: f32, world: &World)
    {
        let mut gfx_api_ref = self.gfx_api.borrow_mut();

        // Process each material initialization/update
        for material in world.query::<Material>()
        {
            let mut material_ref = material.borrow_mut();
            let material_name = material_ref.name.clone();

            // Initialize material if it hasn't been initialized yet
            if !material_ref.has_been_initialized
            {
                // Have to initialize shaders first, otherwise texture assignments can't be done
                gfx_api_ref.init_shaders(
                    &material_ref.vertex_shader_path,
                    &material_ref.fragment_shader_path,
                    &material_ref.name,
                    material_ref.textures.len() as u32,
                    material_ref.lighting_mode,
                );

                // Initialize textures
                let mut texture_data = Vec::new();

                // If material is PBR, check that all textures are present
                // If not present, add default textures
                if material_ref.lighting_mode == LightingMode::PBR
                {
                    Self::add_missing_pbr_textures(&mut material_ref.textures);
                }

                for texture in material_ref.textures.iter()
                {
                    // Get default image data when initializing default textures
                    // Otherwise just load the image from path
                    let image_data = match texture.filepath.as_str()
                    {
                        DEFAULT_COLOR_TEXTURE_PATH => ImageData::default_white(),
                        DEFAULT_ORM_TEXTURE_PATH => ImageData::default_orm(),
                        DEFAULT_NORMAL_TEXTURE_PATH => ImageData::default_normal(),
                        _ => load_image(&texture.filepath),
                    };

                    texture_data.push((image_data, texture));
                }

                gfx_api_ref.init_textures(texture_data, &material_ref.name);

                // Update PBR parameters if material is PBR
                if material_ref.lighting_mode == LightingMode::PBR
                {
                    // Add default PBR parameters if parameters have not been added
                    if material_ref.pbr_parameters.is_none()
                    {
                        material_ref.pbr_parameters = Some(PbrParameters::default());
                    }

                    let parameters = material_ref.pbr_parameters.as_mut().unwrap();
                    gfx_api_ref.update_pbr_parameters(&material_name, parameters);

                    parameters.was_updated = false;
                }

                material_ref.has_been_initialized = true;
            }

            // Update PBR materials if they were changed
            if material_ref.lighting_mode == LightingMode::PBR
                && material_ref.pbr_parameters.is_some()
            {
                let parameters = material_ref.pbr_parameters.as_mut().unwrap();
                if parameters.was_updated
                {
                    gfx_api_ref.update_pbr_parameters(&material_name, parameters);

                    parameters.was_updated = false;
                }
            }
        }
    }
}
