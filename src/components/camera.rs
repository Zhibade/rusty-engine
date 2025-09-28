///! Camera component implementation

use glam::Mat4;

/// Represents a camera in the world with transform and FOV values
pub struct Camera
{
    pub transform: Mat4,
    pub fov: f32,
}

/// Camera utility functions
impl Camera
{
    /// Get view & perspective projection matrices from the given camera
    pub fn get_view_projection_matrices(
        camera: &Camera,
        window_size: [f32; 2],
        near_far_distance: [f32; 2]
    ) -> ([[f32; 4]; 4], [[f32; 4]; 4])
    {
        let (_, _, translation) = camera.transform.to_scale_rotation_translation();

        let view_matrix = Mat4::look_at_lh(
            translation,
            camera.transform.z_axis.truncate(),
            camera.transform.y_axis.truncate(),
        );

        let projection_matrix = Mat4::perspective_lh(
            f32::to_radians(camera.fov),
            window_size[0] / window_size[1],
            near_far_distance[0],
            near_far_distance[1]
        );

        (view_matrix.to_cols_array_2d(), projection_matrix.to_cols_array_2d())
    }
}

/// Default trait implementation
impl Default for Camera
{
    /// Creates a camera at the origin with an FOV of 60
    fn default() -> Self
    {
        Camera { transform: Mat4::IDENTITY, fov: 60.0, }
    }
}