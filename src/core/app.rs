//! Window application implementation

use crate::components::camera::Camera;
use crate::components::transform::Transform;
use crate::core::world::World;
use crate::gfx_api::GfxApi;
use crate::loaders::mesh_loader;
use crate::systems::camera_rendering::CameraRenderingSystem;
use crate::systems::mesh_rendering::MeshRenderingSystem;
use crate::systems::mesh_rotator::MeshRotatorSystem;
use glam::Vec3;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Window application
#[derive(Default)]
pub struct App
{
    gfx_api: Option<Rc<RefCell<GfxApi>>>,
    window: Option<Arc<Window>>,
    world: Option<World>,
}

impl App
{
    /// Initialize world + default camera and mesh
    fn init_world(&mut self)
    {
        let mut world = World::default();

        // Add camera entity
        let camera_id = world.add_entity();
        let camera_transform = Transform::from_position(Vec3::from_array([0.0, 0.5, -1.5]));

        world.add_component(camera_id, Camera::default());
        world.add_component(camera_id, camera_transform);

        // Add mesh entity
        let mesh = mesh_loader::load_mesh("assets/table.gltf".replace("/", "\\"));

        let mesh_id = world.add_entity();
        world.add_component(mesh_id, mesh);
        world.add_component(mesh_id, Transform::default());

        // Register camera and mesh systems
        let mesh_rendering_system = MeshRenderingSystem::new(self.gfx_api.clone().unwrap());
        let mesh_rotator_system = MeshRotatorSystem::default();
        let camera_rendering_system = CameraRenderingSystem::new(
            self.gfx_api.clone().unwrap(),
            self.window.as_ref().unwrap().inner_size().into(),
        );

        world.register_system(mesh_rendering_system);
        world.register_system(mesh_rotator_system);
        world.register_system(camera_rendering_system);

        self.world = Some(world);
    }

    /// Run window application
    pub fn run(&mut self)
    {
        let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");
        event_loop.set_control_flow(ControlFlow::Poll); // Continuously run event loop since window updates every frame
        event_loop.run_app(self).expect("Failed to open window");
    }
}

/// Winit's API implementation
impl ApplicationHandler for App
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop)
    {
        // Initialize window and graphics API on window startup
        self.window =
            Some(Arc::new(event_loop.create_window(Window::default_attributes()).unwrap()));

        // Initialize the world
        self.gfx_api =
            Some(Rc::new(RefCell::new(GfxApi::new(event_loop, self.window.clone().unwrap()))));

        self.init_world();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    )
    {
        match event
        {
            WindowEvent::CloseRequested =>
            {
                event_loop.exit();
            }

            WindowEvent::RedrawRequested =>
            {
                // Update world
                if let Some(ref mut world) = self.world
                {
                    world.update();
                }

                // Render frame and update window
                if let Some(ref mut gfx_api) = self.gfx_api
                {
                    gfx_api.borrow_mut().render();
                }

                self.window.as_ref().unwrap().request_redraw();
            }

            WindowEvent::Resized(size) =>
            {
                if let Some(ref mut gfx_api) = self.gfx_api
                {
                    gfx_api.borrow_mut().on_window_resized(size);
                }
            }

            _ => (),
        }
    }
}
