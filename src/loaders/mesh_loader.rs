///! Mesh loading utilities
use super::super::components::mesh::Mesh;
use glam::{Vec2, Vec3, Vec4};
use gltf;
use std::path::Path;

const MAX_UV_SETS: u32 = 2;

/// Load mesh data from GLTF file
/// If GLTF file contains many meshes, only the first one found will be loaded
/// #Panics
/// Will panic if filepath does not exist or if GLTF file does not contain any meshes
/// Will also panic if mesh does not contain at least positions, indices, normals, and tangents
pub fn load_mesh(filepath: impl AsRef<Path>) -> Mesh
{
    let resolved_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(filepath);
    let path_str = resolved_path.to_str().unwrap(); // For logging

    #[cfg(debug_assertions)]
    println!("Attempting to load GLTF mesh file: {path_str}");

    // Check if file exists
    match resolved_path.try_exists()
    {
        Ok(f) => f,
        Err(_) => panic!("Filepath does not exist for mesh file: {path_str}"),
    };

    let (gltf_file, buffers, _) = gltf::import(resolved_path.clone()).unwrap();

    // Check if there is a mesh in the imported file
    if gltf_file.meshes().count() == 0
    {
        panic!("Imported GLTF file does not contain any meshes: {path_str}");
    }

    // Get first mesh's data
    let mesh_data = gltf_file
        .meshes()
        .nth(0)
        .unwrap()
        .primitives()
        .nth(0)
        .unwrap();
    let mesh_data_reader = mesh_data.reader(|buffer| Some(&buffers[buffer.index()]));

    let mut mesh = Mesh::default();

    // Positions
    if let Some(positions) = mesh_data_reader.read_positions()
    {
        positions.for_each(|position| {
            mesh.vertices
                .push(Vec3::new(position[0], position[1] * 1.0, position[2]));
        });
    }
    else
    {
        panic!("Imported GLTF file's mesh doesn't have or has corrupted vertices data: {path_str}");
    }

    // Indices
    if let Some(indices) = mesh_data_reader.read_indices()
    {
        mesh.indices = indices.into_u32().collect();
    }
    else
    {
        panic!("Imported GLTF file's mesh doesn't have or has corrupted indices data: {path_str}");
    }

    // Normals
    if let Some(normals) = mesh_data_reader.read_normals()
    {
        normals.for_each(|normal| {
            mesh.normals
                .push(Vec3::new(normal[0], normal[1] * 1.0, normal[2]))
        });
    }
    else
    {
        panic!("Imported GLTF file's mesh doesn't have or has corrupted normals data: {path_str}");
    }

    // Tangents
    // TODO: Calculate using mikktspace when not present
    if let Some(tangents) = mesh_data_reader.read_tangents()
    {
        tangents.for_each(|tangent| {
            mesh.tangents
                .push(Vec3::new(tangent[0], tangent[1] * 1.0, tangent[2]))
        });
    }
    else
    {
        panic!("Imported GLTF file's mesh doesn't have or has corrupted normals data: {path_str}");
    }

    // UV sets (optional)
    // Hard-coded for 2 uv sets only for now
    // If there is no data just set them all to zero
    for index in 0..MAX_UV_SETS
    {
        mesh.uv_sets.push(Vec::new());

        if let Some(uvs) = mesh_data_reader.read_tex_coords(index)
        {
            let f32_uvs = uvs.into_f32();

            f32_uvs.for_each(|uvs| {
                mesh.uv_sets[index as usize].push(Vec2::from_array(uvs));
            });
        }
        else
        {
            mesh_data_reader.read_positions().unwrap().for_each(|_| {
                mesh.uv_sets[index as usize].push(Vec2::ZERO);
            });
        }
    }

    // Colors (optional)
    // Just reading first color set for now
    // Setting them to white when there is no data
    if let Some(colors) = mesh_data_reader.read_colors(0)
    {
        let f32_colors = colors.into_rgba_f32();

        f32_colors.for_each(|color| {
            mesh.colors.push(Vec4::from_array(color));
        });
    }
    else
    {
        mesh_data_reader.read_positions().unwrap().for_each(|_| {
            mesh.colors.push(Vec4::from_array([1.0, 1.0, 1.0, 1.0]));
        });
    }

    mesh
}
