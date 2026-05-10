//! Image/texture loader implementation

use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};
use std::path::Path;

/// Pixel data that can be represented in different formats
pub enum PixelData
{
    SDR(Vec<u8>),
    HDR(Vec<f32>),
}

impl Default for PixelData
{
    fn default() -> Self
    {
        Self::SDR(Vec::new())
    }
}

/// Contains the image data and other metadata
#[derive(Default)]
pub struct ImageData
{
    pub data: PixelData,
    pub size: [u32; 2],
}

impl ImageData
{
    /// Create a 1x1 white SDR image data
    pub fn default_white() -> Self
    {
        ImageData { data: PixelData::SDR(Vec::from([255; 4])), size: [1, 1], ..Self::default() }
    }

    /// Create 1x1 black SDR image data
    pub fn default_black() -> Self
    {
        ImageData { data: PixelData::SDR(Vec::from([0; 4])), size: [1, 1], ..Self::default() }
    }

    /// Create 1x1 fully reflective metallic ORM (occlusion-roughness-metallic) image data
    pub fn default_orm() -> Self
    {
        ImageData {
            data: PixelData::SDR(Vec::from([255, 0, 255, 255])),
            size: [1, 1],
            ..Self::default()
        }
    }

    /// Create 1x1 neutral normal image data
    pub fn default_normal() -> Self
    {
        ImageData {
            data: PixelData::SDR(Vec::from([128, 128, 255, 255])),
            size: [1, 1],
            ..Self::default()
        }
    }

    /// Create a 1x1 white HDR image data
    pub fn default_white_hdr() -> Self
    {
        ImageData { data: PixelData::HDR(Vec::from([1.0; 4])), size: [1, 1], ..Self::default() }
    }

    /// Create a 1x1 black HDR image data
    pub fn default_black_hdr() -> Self
    {
        ImageData { data: PixelData::HDR(Vec::from([0.0; 4])), size: [1, 1], ..Self::default() }
    }
}

/// Attempts to load the image at the given path (only PNG and EXR are supported for now)\
/// Will return the image data and metadata
///
/// If file does not exist or is not valid, a placeholder image data will be returned
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
        Err(_) =>
        {
            let message = format!(
                "Filepath does not exist for image file: {path_str}.\
                Using placeholder image."
            );
            log_error(message.as_str());

            return ImageData::default_black();
        }
    };

    // Check extension and try to load using corresponding loader
    if let Some(extension) = resolved_path.extension()
    {
        if extension == "png"
        {
            load_png(resolved_path).unwrap_or_else(|e| {
                let message = format!(
                    "An error occurred while reading PNG image file:\
                    {path_str} :: {e}. Using placeholder image"
                );
                log_error(message.as_str());

                ImageData::default_black()
            })
        }
        else if extension == "exr"
        {
            load_exr(resolved_path).unwrap_or_else(|e| {
                let message = format!(
                    "An error occurred while reading EXR image file:\
                    {path_str} :: {e}. Using placeholder image"
                );
                log_error(message.as_str());

                ImageData::default_black_hdr()
            })
        }
        else
        {
            let message = format!(
                "Image file needs to be PNG (.png) or EXR (.exr) format:\
                {path_str}. Using placeholder image"
            );
            log_error(message.as_str());

            ImageData::default_black()
        }
    }
    else
    {
        let message = format!(
            "Could not retrieve extension for image file: {path_str}. \
            Using placeholder image"
        );
        log_error(message.as_str());

        ImageData::default_black()
    }
}

/// Attempt to load an EXR file and return its image contents
fn load_exr(filepath: impl AsRef<Path>) -> Result<ImageData, Error>
{
    // Try to load EXR's first layer
    let exr_image = match exr::prelude::read_first_rgba_layer_from_file(
        filepath.as_ref(),
        |resolution, _| {
            let blank_pixel = [0.0, 0.0, 0.0, 0.0];
            let line = vec![blank_pixel; resolution.width()];
            vec![line; resolution.height()]
        },
        |data, position, (r, g, b, a): (f32, f32, f32, f32)| {
            data[position.y()][position.x()] = [r, g, b, a];
        },
    )
    {
        Ok(value) => value,
        Err(e) => return Err(Error::new(ErrorKind::InvalidData, e.to_string())),
    };

    // Extract image data and return
    let flattened_data =
        exr_image.layer_data.channel_data.pixels.into_iter().flatten().flatten().collect();

    Ok(ImageData {
        data: PixelData::HDR(flattened_data),
        size: [exr_image.layer_data.size.0 as u32, exr_image.layer_data.size.1 as u32],
        ..Default::default()
    })
}

/// Attempt to load a PNG file and return its image contents
fn load_png(filepath: impl AsRef<Path>) -> Result<ImageData, Error>
{
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let png_decoder = png::Decoder::new(reader);

    // Read and store file contents and metadata
    let mut png_reader = png_decoder.read_info()?;

    let buffer_size = png_reader.output_buffer_size();
    if buffer_size.is_none()
    {
        let error = Error::new(ErrorKind::OutOfMemory, "Unable to decode PNG image");
        return Err(error);
    }

    let mut png_data = vec![0; buffer_size.unwrap()];
    let png_info = png_reader.next_frame(&mut png_data)?;

    // If image is not in RGBA format, need to manually add the alpha channel
    // Otherwise it won't be loaded correctly by the graphics API
    if png_data.len() != (png_info.width * png_info.height * 4) as usize
    {
        png_data =
            png_data.chunks_exact(3).flat_map(|data| [data[0], data[1], data[2], 255]).collect();
    }

    Ok(ImageData {
        data: PixelData::SDR(png_data),
        size: [png_info.width, png_info.height],
        ..Default::default()
    })
}

/// Log error with the given message
fn log_error(message: &str)
{
    println!("Error :: image_loader :: {message}");
}
