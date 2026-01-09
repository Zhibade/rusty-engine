//! Light rendering system implementation

use crate::components::image_based_light::ImageBasedLight;
use crate::core::world::World;
use crate::graphics::gfx_api::GfxApi;
use crate::loaders::image_loader::load_image;
use crate::systems::system::System;
use std::cell::RefCell;
use std::rc::Rc;

/// Image-based lighting rendering system
pub struct LightRenderingSystem
{
    gfx_api: Rc<RefCell<GfxApi>>,
}

impl LightRenderingSystem
{
    /// Creates a new [LightRenderingSystem] and passes the given graphics API and window size
    pub fn new(gfx_api: Rc<RefCell<GfxApi>>) -> Self
    {
        LightRenderingSystem { gfx_api }
    }
}

impl System for LightRenderingSystem
{
    fn update(&self, _: f32, world: &World)
    {
        let mut gfx_api_mut = self.gfx_api.borrow_mut();
        let components = world.query::<ImageBasedLight>();

        // Only one environment map is supported, so process only the first one
        if !components.is_empty()
        {
            match components.get(0)
            {
                Some(component) =>
                {
                    let mut ibl = component.borrow_mut();

                    // Initialize texture if not initialized
                    if !ibl.already_initialized
                    {
                        let image_data = load_image(&ibl.cubemap);
                        gfx_api_mut.init_environment_map(image_data);

                        ibl.already_initialized = true;
                    }

                    // Update lighting settings
                    gfx_api_mut.update_lighting(ibl.exposure, ibl.angle);
                }

                None => println!(
                    "LightRenderingSystem :: update :: Unable to retrieve first \"
                                 ImageBasedLight component. Skipping update."
                ),
            }
        }
    }
}
