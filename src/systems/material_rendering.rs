//! Material rendering system implementation

use crate::components::material::Material;
use crate::core::world::World;
use crate::graphics::gfx_api::GfxApi;
use crate::loaders::image_loader::load_image;
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
}

impl System for MaterialRenderingSystem
{
    fn update(&self, _: f32, world: &World)
    {
        let mut gfx_api_ref = self.gfx_api.borrow_mut();

        // Initialize shaders and textures in materials
        for material in world.query::<Material>()
        {
            let mut material_ref = material.borrow_mut();
            if !material_ref.has_been_initialized
            {
                // Have to initialize shaders first, otherwise texture assignments can't be done
                gfx_api_ref.init_shaders(
                    &material_ref.vertex_shader_path,
                    &material_ref.fragment_shader_path,
                    &material_ref.name,
                    material_ref.textures.len() as u32,
                    material_ref.is_lit,
                );

                // Initialize textures
                let mut texture_data = Vec::new();

                for texture in material_ref.textures.iter()
                {
                    let image_data = load_image(&texture.filepath);
                    texture_data.push((image_data, texture));
                }

                gfx_api_ref.init_textures(texture_data, &material_ref.name);
                material_ref.has_been_initialized = true;
            }
        }
    }
}
