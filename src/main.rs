//! Rusty engine main entry point

mod app;
mod gfx_api;

fn main()
{
    let mut app = app::App::default();
    app.run();
}