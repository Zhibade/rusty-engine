//! Post process rendering system implementation

use crate::components::post_process::PostProcess;
use crate::core::world::World;
use crate::graphics::gfx_api::GfxApi;
use crate::systems::system::System;
use std::cell::RefCell;
use std::rc::Rc;

/// Initializes post processes for rendering
pub struct PostProcessRenderingSystem
{
    gfx_api: Rc<RefCell<GfxApi>>
}

impl PostProcessRenderingSystem
{
    /// Creates a new [PostProcessRenderingSystem] and passes the given graphics API
    pub fn new(gfx_api: Rc<RefCell<GfxApi>>) -> Self
    {
        PostProcessRenderingSystem { gfx_api }
    }
}

impl System for PostProcessRenderingSystem
{
    fn update(&self, delta_time: f32, world: &World)
    {
        let mut gfx_api_ref = self.gfx_api.borrow_mut();

        for post_process in world.query::<PostProcess>()
        {
            let mut post_process_mut = post_process.borrow_mut();

            if !post_process_mut.has_been_initialized
            {
                gfx_api_ref.init_post_process(
                    post_process_mut.shader_path.as_str(),
                    post_process_mut.priority,
                );

                post_process_mut.has_been_initialized = true;
            }
        }
    }
}
