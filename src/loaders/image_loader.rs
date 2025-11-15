//! Image/texture loader implementation

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Contains the image byte data and other metadata
#[derive(Default)]
pub struct ImageData
{
    pub data: Vec<u8>,
    pub size: [u32; 2],
}

impl ImageData
{
    /// Create a 2x2 white image data
    pub fn default_white() -> Self
    {
        ImageData { data: Vec::from([255; 4]), size: [1, 1] }
    }

    /// Create 2x2 black image data
    pub fn default_black() -> Self
    {
        ImageData { data: Vec::from([0; 4]), size: [1, 1] }
    }
}

/// Attempts to load the image at the given path (only PNG is supported for now)\
/// Will return the PNG image data and metadata
///
/// # Panics
/// Will panic if file does not exist or is not a valid PNG image
pub fn load_image(filepath: impl AsRef<Path>) -> ImageData
{
    let resolved_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(filepath);
    let path_str = String::from(resolved_path.to_str().unwrap());

    #[cfg(debug_assertions)]
    println!("Attempting to load image file: {path_str}");

    // Check if file exists
    match resolved_path.try_exists()
    {
        Ok(f) => f,
        Err(_) => panic!("Filepath does not exist for image file: {path_str}"),
    };

    // Load PNG file
    let file = File::open(resolved_path).unwrap();
    let reader = BufReader::new(file);
    let png_decoder = png::Decoder::new(reader);

    // Read and store file contents and metadata
    let mut png_reader = png_decoder.read_info().unwrap();
    let mut png_data = vec![0; png_reader.output_buffer_size().unwrap()];
    let png_info = png_reader.next_frame(&mut png_data).unwrap();

    // If image is not in RGBA format, need to manually add the alpha channel
    // Otherwise it won't be loaded correctly by the graphics API
    if png_data.len() != (png_info.width * png_info.height * 4) as usize
    {
        png_data =
            png_data.chunks_exact(3).flat_map(|data| [data[0], data[1], data[2], 255]).collect();
    }

    ImageData { data: png_data, size: [png_info.width, png_info.height] }
}
