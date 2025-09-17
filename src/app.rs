//! Window application implementation

use super::gfx_api::GfxApi;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop:: {ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Window application
#[derive(Default)]
pub struct App
{
    window: Option<Arc<Window>>,
    gfx_api: Option<GfxApi>,
}

impl App
{
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
        self.window = Some(Arc::new(event_loop.create_window(Window::default_attributes()).unwrap()));
        self.gfx_api = Some(GfxApi::new(event_loop, self.window.clone().unwrap()));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent)
    {
        match event
        {
            WindowEvent::CloseRequested =>
                {
                    event_loop.exit();
                },

            WindowEvent::RedrawRequested =>
                {
                    // Render frame and update window
                    self.gfx_api.as_mut().unwrap().render();
                    self.window.as_ref().unwrap().request_redraw();
                }

            WindowEvent::Resized(size) =>
                {
                    self.gfx_api.as_mut().unwrap().on_window_resized(size);
                }

            _ => (),
        }
    }
}

