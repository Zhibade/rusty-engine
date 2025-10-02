//! Window application implementation

use super::components::camera::Camera;
use super::gfx_api::GfxApi;
use super::loaders::mesh_loader;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Window application
#[derive(Default)]
pub struct App {
    camera: Camera,
    gfx_api: Option<GfxApi>,
    window: Option<Arc<Window>>,
}

impl App {
    /// Run window application
    pub fn run(&mut self) {
        let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");
        event_loop.set_control_flow(ControlFlow::Poll); // Continuously run event loop since window updates every frame
        event_loop.run_app(self).expect("Failed to open window");
    }
}

/// Winit's API implementation
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Get mesh data first. No need to initialize everything if mesh data is invalid
        let mesh = mesh_loader::load_mesh("assets/table.gltf".replace("/", "\\"));

        // Initialize window and graphics API on window startup
        self.window = Some(Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        ));
        self.gfx_api = Some(GfxApi::new(event_loop, self.window.clone().unwrap(), mesh));
        self.camera = Camera {
            transform: Mat4::from_translation(Vec3::from_array([0.0, 0.5, -1.5])),
            ..Default::default()
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::RedrawRequested => {
                // Send camera updated transform to graphics API
                let (view_matrix, projection_matrix) = Camera::get_view_projection_matrices(
                    &self.camera,
                    self.window.as_ref().unwrap().inner_size().into(),
                    [0.0, 10000.0],
                );

                self.gfx_api
                    .as_mut()
                    .unwrap()
                    .update_camera(view_matrix, projection_matrix);

                // Render frame and update window
                self.gfx_api.as_mut().unwrap().render();
                self.window.as_ref().unwrap().request_redraw();
            }

            WindowEvent::Resized(size) => {
                self.gfx_api.as_mut().unwrap().on_window_resized(size);
            }

            _ => (),
        }
    }
}
