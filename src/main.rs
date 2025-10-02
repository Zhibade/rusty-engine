//! Rusty engine main entry point

mod app;
mod components;
mod gfx_api;
mod loaders;

fn main() {
    let mut app = app::App::default();
    app.run();
}
