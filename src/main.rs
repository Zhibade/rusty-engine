//! Rusty engine main entry point

mod components;
mod core;
mod graphics;
mod loaders;
mod systems;

// Includes
use crate::core::app::App;

// Program's entry point
fn main()
{
    let mut app = App::default();
    app.run();
}
