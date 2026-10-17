//! Render camera system implementation

use crate::components::camera::Camera;
use crate::components::transform::Transform;
use crate::core::world::World;
use crate::graphics::gfx_api::GfxApi;
use crate::systems::system::System;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

pub struct CameraRenderingSystem
{
    gfx_api: Rc<RefCell<GfxApi>>,
    window_size: [f32; 2],
}

impl CameraRenderingSystem
{
    /// Creates a new [CameraRenderingSystem] and passes the given graphics API and window size
    pub fn new(gfx_api: Rc<RefCell<GfxApi>>, window_size: [f32; 2]) -> Self
    {
        CameraRenderingSystem { gfx_api, window_size }
    }
}

impl System for CameraRenderingSystem
{
    /// Will set and update the first found active render camera\
    /// If multiple active cameras are found, it will set all other cameras as inactive
    fn update(&self, _: f32, world: &World)
    {
        let mut gfx_api_mut = self.gfx_api.borrow_mut();
        let mut found_active = false;

        let component_pairs = world.query_pair::<Camera, Transform>();

        for (camera, transform) in component_pairs
        {
            let transform_ref = transform.borrow();
            let mut camera_mut = camera.borrow_mut();

            if camera_mut.is_active_camera
            {
                // If already found another active camera, just set it to inactive
                if found_active
                {
                    camera_mut.is_active_camera = false;
                    continue;
                }

                // Update camera matrices on graphics API
                let (view_matrix, projection_matrix) = Camera::get_view_projection_matrices(
                    camera_mut.deref(),
                    transform_ref.deref(),
                    self.window_size,
                    [0.01, 100.0],
                );

                gfx_api_mut.update_camera(
                    transform_ref.matrix.to_scale_rotation_translation().2.to_array(),
                    view_matrix,
                    projection_matrix,
                );

                found_active = true;
            }
        }
    }
}
