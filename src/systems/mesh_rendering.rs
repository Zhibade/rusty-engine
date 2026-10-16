//! Mesh rendering system implementation

use crate::components::material::Material;
use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::core::world::World;
use crate::graphics::gfx_api::GfxApi;
use crate::systems::system::System;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

/// Mesh rendering system
///
/// Responsible for passing mesh data to graphics API
pub struct MeshRenderingSystem
{
    gfx_api: Rc<RefCell<GfxApi>>,
}

impl MeshRenderingSystem
{
    /// Creates a new [MeshRenderingSystem] and passes the given graphics API
    pub fn new(gfx_api: Rc<RefCell<GfxApi>>) -> Self
    {
        MeshRenderingSystem { gfx_api }
    }
}

impl System for MeshRenderingSystem
{
    /// For newly added [Mesh] + [Transform] component pairs, it will send mesh and transform data
    /// to the graphics API for initialization.
    ///
    /// For component pairs that have already been initialized, it will send the updated transform
    /// data to the graphics API if there have been any changes to it
    fn update(&self, _: f32, world: &World)
    {
        let mut gfx_api_ref = self.gfx_api.borrow_mut();

        // Get mesh, transform, and material components
        for (mesh_rc, transform_rc, material_rc) in world.query_trio::<Mesh, Transform, Material>()
        {
            let mut mesh = mesh_rc.borrow_mut();
            let transform = transform_rc.borrow();
            let material = material_rc.borrow();

            // Initialize mesh if it is its first update
            // Otherwise just update transform if there have been changes
            if !mesh.has_been_initialized
            {
                mesh.instance_id = gfx_api_ref.init_mesh(
                    mesh.deref(),
                    transform.matrix.to_cols_array_2d(),
                    &material.name,
                );
                mesh.has_been_initialized = true;
            }
            else if transform.has_been_changed
            {
                gfx_api_ref.update_mesh_transform(
                    &mesh.filepath,
                    &material.name,
                    mesh.instance_id,
                    transform.matrix.to_cols_array_2d(),
                );
            }
        }
    }
}
