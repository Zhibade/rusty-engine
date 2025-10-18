//! Rusty engine main entry point

mod app;
mod components;
mod gfx_api;
mod loaders;
mod systems;
mod world;

fn main()
{
    let mut app = app::App::default();
    app.run();
}
