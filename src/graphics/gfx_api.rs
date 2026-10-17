//! Graphics API that will be used to render onto the application's window

use crate::components::material::{LightingMode, PbrParameters, Texture, TextureUsage};
use crate::components::mesh::Mesh;
use crate::loaders::image_loader::{ImageData, PixelData};
use std::cmp;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::mem::{self, size_of};
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BlitImageInfo, CommandBufferExecFuture, CommandBufferUsage,
    CopyBufferInfo, CopyBufferToImageInfo, ImageBlit, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::{
    DescriptorSetAllocator, StandardDescriptorSetAllocator,
    StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::{ClearValue, Format, FormatFeatures};
use vulkano::image::sampler::{
    Filter, LOD_CLAMP_NONE, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    Image, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageSubresourceLayers,
    ImageSubresourceRange, ImageTiling, ImageType, ImageUsage,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::library::VulkanLibrary;
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{
    ComputePipeline, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{
    AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer,
    FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass, SubpassDependency,
    SubpassDescription,
};
use vulkano::shader::{EntryPoint, ShaderModule, ShaderModuleCreateInfo, ShaderStages};
use vulkano::swapchain::{
    self, CompositeAlpha, PresentFuture, Surface, Swapchain, SwapchainAcquireFuture,
    SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture, NowFuture};
use vulkano::sync::{self, AccessFlags, GpuFuture, PipelineStages};
use vulkano::{DeviceSize, Validated, Version, VulkanError};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

// ------
// Constants
// ------

const VIEW_PROJ_DESCRIPTOR_SET_LAYOUT_INDEX: usize = 0;
const FRAGMENT_SHADER_EXTRA_DATA_SET_LAYOUT_INDEX: usize = 1;
const TEXTURES_DESCRIPTOR_SET_LAYOUT_INDEX: usize = 2;
const LIGHTING_DESCRIPTOR_SET_LAYOUT_INDEX: usize = 3;
const PBR_PARAMS_DESCRIPTOR_SET_LAYOUT_INDEX: usize = 4;

const CUBEMAP_CONVERSION_SHADER_PATH: &str = "src/graphics/shaders/cubemap_conversion_comp.spv";
const IRRADIANCE_MAP_SUFFIX: &str = "irradiance-map";
const IRRADIANCE_MAP_CONVOLUTION_SHADER_PATH: &str =
    "src/graphics/shaders/irradiance_map_convolution_comp.spv";
const PREFILTERED_MAP_SUFFIX: &str = "prefiltered-map";
const PREFILTERED_MAP_CONVOLUTION_SHADER_PATH: &str =
    "src/graphics/shaders/prefiltered_map_convolution_comp.spv";
const BRDF_INTEGRATION_MAP_SHADER_PATH: &str = "src/graphics/shaders/brdf_integration_comp.spv";

const POST_PROCESS_VERTEX_SHADER_PATH: &str = "src/graphics/shaders/post_process_vert.spv";

const FALLBACK_MATERIAL_PIPELINE_NAME: &str = "Fallback";
const FALLBACK_VERTEX_SHADER_PATH: &str = "src/graphics/shaders/base_vert.spv";
const FALLBACK_FRAGMENT_SHADER_PATH: &str = "src/graphics/shaders/fallback_frag.spv";
const FALLBACK_POST_PROCESS_SHADER_PATH: &str =
    "src/graphics/shaders/post_process_fallback_frag.spv";

const HDR_FORMAT: Format = Format::A2B10G10R10_UNORM_PACK32;
const SDR_FORMAT: Format = Format::R8G8B8A8_SRGB;

// ------
// Aliases
// ------

type SubpassAttachments = Vec<Arc<ImageView>>;

// ------
// Internal enums
// ------

/// Method used when dispatching a compute shader
#[derive(Copy, Clone, Eq, PartialEq)]
enum ComputeDispatchMethod
{
    Once,
    OncePerMipmap,
    OncePerMipmapAlwaysUseFirstMipmapAsInput,
}

// ------
// Internal structs
// ------

/// Cubemap processing compute pipeline data
#[derive(Debug)]
struct CubemapComputePipelineData
{
    pipeline: Arc<ComputePipeline>,
    max_mipmap_num: u32, // Maximum number of mipmaps that will be output from the shader

    // Whether the shader uses image and mipmap metadata push constants
    uses_metadata_constants: bool,
}

/// Environment texture data needed for PBR image-based lighting
struct EnvironmentTextureData
{
    environment_map: TextureData,
    irradiance_map: TextureData,
    prefiltered_map: TextureData,
    brdf_integration_map: TextureData,
}

/// Shader/material pipeline and texture assignments data
struct MaterialPipelineData
{
    vert_shader_path: String, // Necessary for recreating pipeline when render pass changes
    frag_shader_path: String, // Necessary for recreating pipeline when render pass changes
    graphics_pipeline: Arc<GraphicsPipeline>,
    lighting_mode: LightingMode,
    pbr_parameters_data: Option<PbrParametersData>,
    texture_assignments: HashMap<TextureUsage, String>,
    textures_descriptor_set: Option<Arc<DescriptorSet>>,
}

/// Post process pipeline and intermediate render target(s) descriptor set
struct PostProcessPipelineData
{
    shader_path: String, // Necessary for recreating pipeline when render pass changes
    shader_data: PostProcessShaderData,
    priority: u32,
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    attachments_image_views: Vec<SubpassAttachments>, // One set per swapchain image
    attachments_descriptor_sets: Vec<Arc<DescriptorSet>>, // One per swapchain image
}

/// PBR parameters data after uploading to the GPU
struct PbrParametersData
{
    pbr_parameters: PbrParametersBufferData,
    pbr_parameters_buffer: Subbuffer<PbrParametersBufferData>,
    pbr_parameters_descriptor_set: Arc<DescriptorSet>,
    updated_parameters: bool,
}

/// Render mesh data after uploading to the GPU
struct RenderMeshData
{
    vertex_buffer: Subbuffer<[VertexData]>,
    index_buffer: Subbuffer<[u32]>,

    // TODO: Might be better to use indexes/GUID instead of material name?
    transform_data_per_pipeline_id: HashMap<String, Vec<ObjectTransformData>>,
}

/// Texture data after uploading to the GPU
#[derive(Clone)]
struct TextureData
{
    image_view: Arc<ImageView>,
    sampler: Arc<Sampler>,
}

/// Extra fragment shader shared data after uploading to the GPU
struct FragmentShaderExtraData
{
    data: FragmentShaderExtraDataContents,
    buffer: Subbuffer<FragmentShaderExtraDataContents>,
    descriptor_set: Arc<DescriptorSet>,
    updated: bool,
}

// ------
// Buffer structs
// ------

/// Object's transform data (passed as a push constant)
#[derive(BufferContents, Clone, Default)]
#[repr(C)]
struct ObjectTransformData
{
    matrix: [[f32; 4]; 4],
}

/// Model vertex buffer data
/// Support for two UV sets only for now
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct VertexData
{
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],

    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],

    #[format(R32G32B32_SFLOAT)]
    tangent: [f32; 3],

    #[format(R32G32_SFLOAT)]
    uv0: [f32; 2],

    #[format(R32G32_SFLOAT)]
    uv1: [f32; 2],

    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

/// View + perspective projection matrices (passed as a uniform buffer)
#[derive(BufferContents, Clone, Default)]
#[repr(C)]
struct ViewProjectionData
{
    view_matrix: [[f32; 4]; 4],
    perspective_matrix: [[f32; 4]; 4],
}

/// Metadata constants used in environment map processing compute shaders
#[derive(BufferContents)]
#[repr(C)]
struct EnvironmentMapComputeConstants
{
    input_image_size: [u32; 2],
    current_mipmap: u32,
    total_mipmap_num: u32,
}

/// Image-based lighting data
#[derive(BufferContents, Clone)]
#[repr(C)]
struct EnvironmentMapData
{
    exposure: f32,
    angle: f32,
}

/// PBR parameters buffer contents
#[derive(BufferContents, Clone)]
#[repr(C)]
struct PbrParametersBufferData
{
    color_tint: [f32; 4],
    occlusion_strength: f32,
    roughness_strength: f32,
    metallic_strength: f32,
    normal_strength: f32,
}

impl Default for PbrParametersBufferData
{
    fn default() -> Self
    {
        PbrParametersBufferData {
            color_tint: [1.0, 1.0, 1.0, 1.0],
            occlusion_strength: 1.0,
            roughness_strength: 1.0,
            metallic_strength: 1.0,
            normal_strength: 1.0,
        }
    }
}

/// Extra shared data for use in fragment shaders
#[derive(BufferContents, Clone)]
#[repr(C)]
struct FragmentShaderExtraDataContents
{
    camera_pos: [f32; 3],
}

/// Post process shader data (used as a push constant)
#[derive(BufferContents, Clone, Default)]
#[repr(C)]
struct PostProcessShaderData
{
    strength: f32,
}

// ------
// GfxApi implementation
// ------

/// Graphics API interface
/// Only Vulkan is supported at the moment
pub struct GfxApi
{
    // Base objects/data needed for drawing
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
    should_recreate_render_pass: bool,
    shader_entry_points: HashMap<String, EntryPoint>, // Used for recreating graphics pipelines

    // Unique name -> Material pipeline data
    material_pipeline_data: HashMap<String, MaterialPipelineData>,
    post_processing_pipelines: Vec<PostProcessPipelineData>, // Ordered post process subpasses

    cubemap_conversion_pipeline_data: CubemapComputePipelineData,
    irradiance_map_pipeline_data: CubemapComputePipelineData,
    prefiltered_map_pipeline_data: CubemapComputePipelineData,
    framebuffers: Vec<Arc<Framebuffer>>,
    depth_stencil_format: Format,

    // Memory allocators
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    // Shader data
    render_meshes_data: HashMap<String, RenderMeshData>,
    texture_data: HashMap<String, TextureData>,

    environment_map_data: EnvironmentMapData,
    environment_map_data_buffer: Subbuffer<EnvironmentMapData>,
    environment_map_descriptor_set: Arc<DescriptorSet>,
    environment_texture_data: EnvironmentTextureData,
    updated_environment_map_data: bool,
    updated_environment_map_texture: bool,

    view_projection_data: ViewProjectionData, // Current camera (view-projection) data
    view_projection_data_buffer: Subbuffer<ViewProjectionData>,
    view_projection_data_descriptor_set: Arc<DescriptorSet>,

    fragment_shader_extra_data: FragmentShaderExtraData,

    // Data related to presenting onto the surface each frame
    should_recreate_swapchain: bool,
    window_size: [u32; 2],
    previous_frame_end: Option<
        FenceSignalFuture<
            PresentFuture<CommandBufferExecFuture<JoinFuture<NowFuture, SwapchainAcquireFuture>>>,
        >,
    >,
    // ^ Need this ridiculous nested type to be able to use FenceSignalFuture's API
    // TODO: Might be able to improve it through a different API?
}

impl GfxApi
{
    // --- Associated Functions ---

    /// Create and initialize a new graphics API instance
    /// API will be ready after being created, only need to call [render](GfxApi::render) method
    pub fn new(event_loop: &ActiveEventLoop, window: Arc<Window>) -> Self
    {
        // Init Vulkan library
        let library = VulkanLibrary::new().expect("Vulkan library creation failed");

        // Find required extensions for winit window and initialize instance
        let required_extensions = Surface::required_extensions(event_loop)
            .expect("Failed to retrieve required extensions for Vulkan from winit window");

        #[cfg(debug_assertions)]
        Self::log_required_extensions(&required_extensions);

        let instance_create_info = InstanceCreateInfo {
            application_name: Some(String::from("Rusty Engine")),
            application_version: Version::major_minor(1, 0),
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            enabled_layers: Self::get_enabled_layers(),
            ..Default::default()
        };

        let instance =
            Instance::new(library, instance_create_info).expect("Failed to create Vulkan instance");

        // Find the best physical device candidate and create a logical device for it
        let device_extensions =
            DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::empty() };

        let (physical_device, graphics_queue_index, compute_queue_index) =
            GfxApi::get_best_physical_device_and_queue(&instance, &event_loop, &device_extensions);

        let physical_device_properties = physical_device.properties();

        #[cfg(debug_assertions)]
        println!(
            "Initializing logical device using {} :: {}",
            physical_device_properties.device_name,
            Self::get_device_type_name(&physical_device_properties.device_type)
        );

        // Create QueueCreateInfos
        let queue_create_infos = {
            let mut create_infos = Vec::from([QueueCreateInfo {
                queue_family_index: graphics_queue_index,
                ..Default::default()
            }]);

            // Create a separate QueueCreateInfo for compute if it is different from graphics
            if graphics_queue_index != compute_queue_index
            {
                create_infos.push(QueueCreateInfo {
                    queue_family_index: compute_queue_index,
                    ..Default::default()
                });
            }

            create_infos
        };

        // Initialize logical device from physical device and store queue for later
        let device_create_info = DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos,
            ..Default::default()
        };

        let (device, mut queues) = Device::new(physical_device, device_create_info)
            .expect("Failed to create logical device");

        let graphics_queue = queues
            .find(|queue| queue.queue_family_index() == graphics_queue_index)
            .expect("Could not find graphics queue in logical device");

        let compute_queue = queues
            .find(|queue| queue.queue_family_index() == compute_queue_index)
            .expect("Could not find compute queue in logical device");

        // Create surface and swapchain
        let surface = Surface::from_window(instance.clone(), window.clone())
            .expect("Failed  to create surface");

        let min_image_count = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap()
            .min_image_count
            .max(2); // Need at least two for fullscreen

        let (image_format, _) =
            device.physical_device().surface_formats(&surface, Default::default()).unwrap()[0];

        let swapchain_create_info = SwapchainCreateInfo {
            min_image_count,
            image_format,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: CompositeAlpha::Opaque, // No need for making it a transparent window
            ..Default::default()
        };

        let (swapchain, swapchain_images) =
            Swapchain::new(device.clone(), surface, swapchain_create_info)
                .expect("Failed to create swapchain");

        // Find best depth buffer format
        let depth_stencil_format =
            Self::get_best_supported_depth_buffer_format(device.physical_device());

        #[cfg(debug_assertions)]
        Self::log_depth_stencil_format(&depth_stencil_format);

        // Create single pass render pass by default (will be recreated later if
        // post processes are added)
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: depth_stencil_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: { depth_stencil },
            },
        )
        .unwrap();

        // Create viewport
        let viewport = Viewport { extent: window.inner_size().into(), ..Default::default() };

        // Create fallback shaders' graphics pipeline
        let mut shader_entry_points = HashMap::new();

        let vert_shader_entry_point = Self::get_shader_entry_point(
            &device,
            FALLBACK_VERTEX_SHADER_PATH,
            &mut shader_entry_points,
        );

        let frag_shader_entry_point = Self::get_shader_entry_point(
            &device,
            FALLBACK_FRAGMENT_SHADER_PATH,
            &mut shader_entry_points,
        );

        let fallback_graphics_pipeline = Self::create_material_graphics_pipeline(
            &device,
            &swapchain,
            &render_pass,
            vert_shader_entry_point,
            frag_shader_entry_point,
            0,
            LightingMode::Unlit,
        );

        // Create cubemap processing compute pipelines
        let [
            cubemap_conversion_pipeline_data,
            irradiance_map_pipeline_data,
            prefiltered_map_pipeline_data,
        ] = Self::create_cubemap_processing_pipelines(
            &device,
            vec![
                (CUBEMAP_CONVERSION_SHADER_PATH, 1, false),
                (IRRADIANCE_MAP_CONVOLUTION_SHADER_PATH, 1, false),
                (PREFILTERED_MAP_CONVOLUTION_SHADER_PATH, 1, true),
            ],
        )
        .try_into()
        .expect("Failed to unpack cubemap compute pipelines vector");

        // Create memory allocators
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let framebuffers = Self::create_framebuffers(
            memory_allocator.clone(),
            &swapchain_images,
            &render_pass,
            &Vec::new(),
            depth_stencil_format,
        );

        // Allocate placeholder environment map data
        let environment_map_data = EnvironmentMapData { exposure: 1.0, angle: 0.0 };
        let environment_map_data_buffer =
            Self::allocate_data_buffer(memory_allocator.clone(), environment_map_data.clone());

        let (env_map_texture_data, irradiance_map_texture_data, prefiltered_map_texture_data) =
            Self::allocate_environment_map_samplers(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                device.clone(),
                graphics_queue.clone(),
                compute_queue.clone(),
                &cubemap_conversion_pipeline_data,
                &irradiance_map_pipeline_data,
                &prefiltered_map_pipeline_data,
                ImageData::default_black_hdr(),
            );

        // Pre-calculate BRDF integration map (only needs to be done once)
        let brdf_integration_map_texture_data = Self::generate_brdf_integration_map(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            compute_queue.clone(),
            [512, 512],
        );

        let environment_texture_data = EnvironmentTextureData {
            environment_map: env_map_texture_data,
            irradiance_map: irradiance_map_texture_data,
            prefiltered_map: prefiltered_map_texture_data,
            brdf_integration_map: brdf_integration_map_texture_data,
        };

        let environment_map_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            Self::get_environment_map_data_layout(device.clone()),
            [
                WriteDescriptorSet::buffer(0, environment_map_data_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    environment_texture_data.irradiance_map.image_view.clone(),
                    environment_texture_data.irradiance_map.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    environment_texture_data.prefiltered_map.image_view.clone(),
                    environment_texture_data.prefiltered_map.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    3,
                    environment_texture_data.brdf_integration_map.image_view.clone(),
                    environment_texture_data.brdf_integration_map.sampler.clone(),
                ),
            ],
            [],
        )
        .expect("Failed to create default environment map descriptor set");

        // Allocate view and perspective projection data
        let view_projection_data = ViewProjectionData::default();
        let view_perspective_data_buffer =
            Self::allocate_data_buffer(memory_allocator.clone(), view_projection_data.clone());

        let view_perspective_descriptor_set_layout = fallback_graphics_pipeline
            .layout()
            .set_layouts()
            .get(VIEW_PROJ_DESCRIPTOR_SET_LAYOUT_INDEX)
            .unwrap()
            .clone();

        let view_perspective_data_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            view_perspective_descriptor_set_layout,
            [WriteDescriptorSet::buffer(0, view_perspective_data_buffer.clone())],
            [],
        )
        .unwrap();

        // Allocate extra fragment shader shared data
        let fragment_shader_extra_data = {
            let data = FragmentShaderExtraDataContents { camera_pos: [0.0, 0.0, 0.0] };

            let buffer = Self::allocate_data_buffer(memory_allocator.clone(), data.clone());

            let set_layout = fallback_graphics_pipeline
                .layout()
                .set_layouts()
                .get(FRAGMENT_SHADER_EXTRA_DATA_SET_LAYOUT_INDEX)
                .expect(
                    "Failed to get fragment shaders extra data set layout \
                              from fallback graphics pipeline",
                )
                .clone();

            let descriptor_set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                set_layout,
                [WriteDescriptorSet::buffer(0, buffer.clone())],
                [],
            )
            .expect(
                "Failed to create descriptor set layout for fragment shaders' extra shared data",
            );

            FragmentShaderExtraData { data, buffer, descriptor_set, updated: false }
        };

        // Register fallback pipeline data
        let mut pipeline_data = HashMap::new();
        pipeline_data.insert(
            FALLBACK_MATERIAL_PIPELINE_NAME.to_string(),
            MaterialPipelineData {
                vert_shader_path: String::from(FALLBACK_VERTEX_SHADER_PATH),
                frag_shader_path: String::from(FALLBACK_FRAGMENT_SHADER_PATH),
                graphics_pipeline: fallback_graphics_pipeline,
                lighting_mode: LightingMode::Unlit,
                pbr_parameters_data: None,
                texture_assignments: HashMap::new(),
                textures_descriptor_set: None,
            },
        );

        GfxApi {
            // Base objects/data needed for drawing
            device,
            graphics_queue,
            compute_queue,
            swapchain,
            viewport,
            render_pass,
            should_recreate_render_pass: false,

            shader_entry_points,
            material_pipeline_data: pipeline_data,
            post_processing_pipelines: Vec::new(),
            cubemap_conversion_pipeline_data,
            irradiance_map_pipeline_data,
            prefiltered_map_pipeline_data,
            framebuffers,
            depth_stencil_format,

            // Memory allocators
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,

            // Shader data
            environment_map_data,
            environment_map_data_buffer,
            environment_map_descriptor_set,
            environment_texture_data,
            updated_environment_map_data: false,
            updated_environment_map_texture: false,

            fragment_shader_extra_data,
            render_meshes_data: HashMap::new(),
            texture_data: HashMap::new(),
            view_projection_data,
            view_projection_data_buffer: view_perspective_data_buffer,
            view_projection_data_descriptor_set: view_perspective_data_descriptor_set,

            // Data related to presenting onto the surface each frame
            should_recreate_swapchain: false,
            window_size: window.inner_size().into(),
            previous_frame_end: None,
        }
    }

    /// Find the best physical device that supports required extensions
    /// Physical device, graphics queue index, and compute queue index are returned
    fn get_best_physical_device_and_queue(
        instance: &Arc<Instance>,
        event_loop: &ActiveEventLoop,
        device_extensions: &DeviceExtensions,
    ) -> (Arc<PhysicalDevice>, u32, u32)
    {
        let (physical_device, graphics_queue_family_index, compute_queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("Failed to retrieve available physical devices")
            .filter_map(|device| {
                // Check if it supports required extensions and queue families
                if device.supported_extensions().contains(&device_extensions)
                {
                    let mut graphics_queue_index: Option<u32> = None;
                    let mut compute_queue_index: Option<u32> = None;
                    let mut fallback_compute_queue_index: Option<u32>;

                    let queue_iter = device.queue_family_properties().iter().enumerate();
                    let queue_num = queue_iter.len();

                    for (i, queue) in queue_iter
                    {
                        // Find first valid graphics queue
                        if graphics_queue_index.is_none()
                        {
                            if queue.queue_flags.intersects(QueueFlags::GRAPHICS)
                                && device.presentation_support(i as u32, event_loop).unwrap()
                            {
                                graphics_queue_index = Some(i as u32);
                            }
                        }

                        // Find best compute queue index
                        if compute_queue_index.is_none()
                        {
                            if queue.queue_flags.intersects(QueueFlags::COMPUTE)
                            {
                                let compute_index = i as u32;
                                fallback_compute_queue_index = Some(compute_index);

                                // A separate queue from graphics is better, so prioritize it if found
                                if let Some(graphics_index) = graphics_queue_index
                                {
                                    if compute_index != graphics_index
                                    {
                                        compute_queue_index = Some(compute_index);
                                    }
                                }

                                // Use the same as graphics if no other compute queue is found
                                if queue_num == i + 1 && compute_queue_index.is_none()
                                {
                                    compute_queue_index = fallback_compute_queue_index;
                                }
                            }
                        }

                        if graphics_queue_index.is_some() && compute_queue_index.is_some()
                        {
                            break;
                        }
                    }

                    if graphics_queue_index.is_some() && compute_queue_index.is_some()
                    {
                        Some((device, graphics_queue_index, compute_queue_index))
                    }
                    else
                    {
                        None
                    }
                }
                else
                {
                    None
                }
            })
            .min_by_key(|(device, _, _)| {
                // Find the best candidate based on the priority below
                match device.properties().device_type
                {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("No suitable physical device found");

        (physical_device, graphics_queue_family_index.unwrap(), compute_queue_family_index.unwrap())
    }

    /// Get validation layers to be used depending on build configuration
    fn get_enabled_layers() -> Vec<String>
    {
        if cfg!(debug_assertions)
        {
            return Vec::from([String::from("VK_LAYER_KHRONOS_validation")]);
        }

        Vec::new()
    }

    /// Allocate cubemap textures and convert them from equirectangular projection to a cubemap
    /// texture layout so that they can be read correctly by Vulkan\
    ///
    /// If `is_environment_map` is `true`, image-based lighting maps (irradiance and prefiltered)
    /// will be generated along with the regular cubemap texture
    fn allocate_cubemap_samplers(
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        graphics_queue: Arc<Queue>,
        compute_queue: Arc<Queue>,
        cubemap_conversion_pipeline_data: &CubemapComputePipelineData,
        irradiance_map_pipeline_data: &CubemapComputePipelineData,
        prefiltered_map_pipeline_data: &CubemapComputePipelineData,
        cubemap_texture_data: Vec<(&str, &TextureData)>,
        is_environment_map: bool,
    ) -> HashMap<String, TextureData>
    {
        let mut new_cubemap_data = HashMap::new();

        // Create temporary command buffer for executing cubemap processing shaders
        let mut compute_command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create compute command buffer for cubemap compute processing");

        // Create temporary command buffer for generating mipmaps
        let mut graphics_command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create graphics command buffer for cubemap mipmap generation");

        // Need to create another graphics and compute command buffers to generate source cubemap
        // + mipmaps before irradiance and prefiltered maps generation since they are used in there
        //
        // NOTE: Instead of using mipmaps to prevent "firefly" artifacts,
        // a solution like "Fast Filtering of Reflection Probes" could be used, but this
        // seems to relatively work fine for now.
        let mut extra_compute_command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create extra compute command buffer for cubemap compute processing");

        let mut extra_graphics_command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create extra graphics command buffer for cubemap mipmap generation");

        // Create target cubemap textures
        for (path, texture_data) in cubemap_texture_data
        {
            // If it is an environment map, generate an irradiance map and prefiltered map as well
            if is_environment_map
            {
                // Convert equirectangular texture to cubemap and generate mipmaps
                let cubemap_data = Self::register_cubemap_conversion_commands(
                    memory_allocator.clone(),
                    descriptor_set_allocator.clone(),
                    cubemap_conversion_pipeline_data,
                    &mut extra_compute_command_buffer_builder,
                    texture_data,
                    false,
                    path,
                );

                Self::register_mipmap_generation_commands(
                    &mut extra_graphics_command_buffer_builder,
                    &cubemap_data.image_view.image(),
                );

                // Irradiance map generation (equirectangular) -> convert to cubemap ->
                // mipmap generation
                let irradiance_equirectangular_map_data =
                    Self::register_environment_map_compute_commands(
                        memory_allocator.clone(),
                        descriptor_set_allocator.clone(),
                        irradiance_map_pipeline_data,
                        &mut compute_command_buffer_builder,
                        &cubemap_data,
                        ComputeDispatchMethod::Once,
                        path,
                    );

                let irradiance_map_data = Self::register_cubemap_conversion_commands(
                    memory_allocator.clone(),
                    descriptor_set_allocator.clone(),
                    cubemap_conversion_pipeline_data,
                    &mut compute_command_buffer_builder,
                    &irradiance_equirectangular_map_data,
                    false,
                    path,
                );

                Self::register_mipmap_generation_commands(
                    &mut graphics_command_buffer_builder,
                    irradiance_map_data.image_view.image(),
                );

                let irradiance_map_path = format!("{path}_{IRRADIANCE_MAP_SUFFIX}").to_string();
                new_cubemap_data.insert(irradiance_map_path, irradiance_map_data);

                // Prefiltered map generation (equirectangular) ->
                // cubemap conversion for each mipmap
                let prefiltered_equirectangular_map_data =
                    Self::register_environment_map_compute_commands(
                        memory_allocator.clone(),
                        descriptor_set_allocator.clone(),
                        prefiltered_map_pipeline_data,
                        &mut compute_command_buffer_builder,
                        &cubemap_data,
                        ComputeDispatchMethod::OncePerMipmapAlwaysUseFirstMipmapAsInput,
                        path,
                    );

                let prefiltered_map_data = Self::register_cubemap_conversion_commands(
                    memory_allocator.clone(),
                    descriptor_set_allocator.clone(),
                    cubemap_conversion_pipeline_data,
                    &mut compute_command_buffer_builder,
                    &prefiltered_equirectangular_map_data,
                    true,
                    path,
                );

                let prefiltered_map_path = format!("{path}_{PREFILTERED_MAP_SUFFIX}").to_string();
                new_cubemap_data.insert(prefiltered_map_path, prefiltered_map_data);

                new_cubemap_data.insert(String::from(path), cubemap_data);
            }
            else
            {
                // If it is not an environment map, just register the cubemap conversion +
                // mipmap generation commands on the regular command buffers
                let cubemap_data = Self::register_cubemap_conversion_commands(
                    memory_allocator.clone(),
                    descriptor_set_allocator.clone(),
                    cubemap_conversion_pipeline_data,
                    &mut compute_command_buffer_builder,
                    texture_data,
                    false,
                    path,
                );

                Self::register_mipmap_generation_commands(
                    &mut graphics_command_buffer_builder,
                    &cubemap_data.image_view.image(),
                );

                new_cubemap_data.insert(String::from(path), cubemap_data);
            }
        }

        if is_environment_map
        {
            // Run extra command buffers first before the shared compute and graphics command
            // buffers
            //
            // (Getting "All submitted commands that refer to imageView must have completed
            // execution" error when using fences between queues, so just leaving it as an
            // explicit CPU wait for now)
            extra_compute_command_buffer_builder
                .build()
                .expect("Failed to build extra environment map compute command buffer builder")
                .execute(compute_queue.clone())
                .expect("Failed to execute extra environment map compute command buffer")
                .then_signal_fence_and_flush()
                .expect("Failed to register fence signal for extra environment map compute execute")
                .wait(None)
                .expect("Failed to wait for extra cubemap processing");

            extra_graphics_command_buffer_builder
                .build()
                .expect("Failed to build extra environment map graphics command buffer builder")
                .execute(graphics_queue.clone())
                .expect("Failed to execute extra environment map graphics command buffer")
                .then_signal_fence_and_flush()
                .expect("Failed to register fence signal for extra environment map graphics command buffer execution")
                .wait(None)
                .expect("Failed to wait for extra mipmap generation");

            compute_command_buffer_builder
                .build()
                .expect("Failed to build environment map compute command buffer builder")
                .execute(compute_queue.clone())
                .expect("Failed to execute environment map compute command buffer")
                .then_signal_fence_and_flush()
                .expect("Failed to register fence signal for environment map compute execute")
                .wait(None) // Block until environment maps are processed
                .expect("Failed to wait for cubemap processing");
        }
        else
        {
            // When not generating  all environment maps, just execute compute command buffer as is
            // and wait
            compute_command_buffer_builder
                .build()
                .expect("Failed to build environment map compute command buffer builder")
                .execute(compute_queue.clone())
                .expect("Failed to execute environment map compute command buffer")
                .then_signal_fence_and_flush()
                .expect("Failed to register fence signal for environment map compute execute")
                .wait(None) // Block until environment maps are processed
                .expect("Failed to wait for cubemap processing");
        }

        // Execute graphics command buffer (for mipmap generation) and wait until it is done
        graphics_command_buffer_builder
            .build()
            .expect("Failed to build cubemap mipmap generation graphics command buffer builder")
            .execute(graphics_queue.clone())
            .expect("Failed to execute cubemap mipmap generation command buffer")
            .then_signal_fence_and_flush()
            .expect("Failed to register fence signal for cubemap mipmap generation execute")
            .wait(None) // Block until mipmaps are generated
            .expect("Failed to wait for cubemap mipmap generation's signal");

        new_cubemap_data
    }

    /// Allocate environment map cubemap samplers\
    /// Returned tuple contains the environment map, irradiance map, and prefiltered map's
    /// [TextureData]
    fn allocate_environment_map_samplers(
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        compute_queue: Arc<Queue>,
        cubemap_conversion_pipeline_data: &CubemapComputePipelineData,
        irradiance_map_pipeline_data: &CubemapComputePipelineData,
        prefiltered_map_pipeline_data: &CubemapComputePipelineData,
        image_data: ImageData,
    ) -> (TextureData, TextureData, TextureData)
    {
        let texture_info = Texture {
            filepath: String::from("environment_map"),
            is_cubemap: true,
            ..Texture::default()
        };

        // Allocate source texture first (will be converted to actual cubemap later)
        let allocated_texture_data = Self::allocate_texture_samplers(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            device.clone(),
            graphics_queue.clone(),
            vec![(image_data, &texture_info)],
        );

        let texture_data = allocated_texture_data
            .get(&texture_info.filepath)
            .expect("Failed to retrieve environment map source texture data");

        // Convert texture to actual cubemap and update image and sampler
        let allocated_cubemap_texture_data = Self::allocate_cubemap_samplers(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            graphics_queue.clone(),
            compute_queue.clone(),
            cubemap_conversion_pipeline_data,
            irradiance_map_pipeline_data,
            prefiltered_map_pipeline_data,
            vec![(texture_info.filepath.as_str(), texture_data)],
            true,
        );

        let cubemap_texture_data = allocated_cubemap_texture_data
            .get(&texture_info.filepath)
            .expect("Failed to retrieve environment map cubemap texture data");

        let irradiance_map_texture_data = allocated_cubemap_texture_data
            .get(format!("{}_{IRRADIANCE_MAP_SUFFIX}", &texture_info.filepath).as_str())
            .expect("Failed to retrieve irradiance map cubemap texture data");

        let prefiltered_map_texture_data = allocated_cubemap_texture_data
            .get(format!("{}_{PREFILTERED_MAP_SUFFIX}", &texture_info.filepath).as_str())
            .expect("Failed to retrieve prefiltered map cubemap texture data");

        (
            cubemap_texture_data.clone(),
            irradiance_map_texture_data.clone(),
            prefiltered_map_texture_data.clone(),
        )
    }

    /// Allocate multiple textures and their samplers using the given [ImageData] and [Texture] data
    fn allocate_texture_samplers(
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        image_data: Vec<(ImageData, &Texture)>,
    ) -> HashMap<String, TextureData>
    {
        let mut new_texture_data = HashMap::new();

        // Create temporary command buffer for copying textures to GPU
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        for (image_data, texture) in image_data
        {
            // Create a temporary staging buffer for sending image data to GPU
            let staging_buffer_create_info =
                BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() };

            let staging_buffer_alloc_create_info = AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            // Create GPU image buffer
            let image_format = Self::get_image_format_from_image_data(&image_data, texture);

            let mipmap_num = {
                // Cubemaps are generated by a compute shader,
                // so don't need mipmaps until after they are generated
                if texture.is_cubemap
                {
                    1
                }
                else
                {
                    let largest_side = cmp::max(image_data.size[0], image_data.size[1]);
                    if largest_side < 2 { 1 } else { largest_side.ilog2() }
                }
            };

            let mut image_usage = ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED;
            if mipmap_num > 1
            {
                // Only set as transfer source when generating mipmaps
                image_usage = image_usage | ImageUsage::TRANSFER_SRC;
            }

            let image_create_info = ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: image_format,
                extent: [image_data.size[0], image_data.size[1], 1],
                mip_levels: mipmap_num,
                usage: image_usage,
                ..Default::default()
            };

            let image = Image::new(
                memory_allocator.clone(),
                image_create_info,
                AllocationCreateInfo::default(),
            )
            .expect(format!("Failed to create image for {}", texture.filepath).as_str());

            // Create image view, sampler, and then add to texture data collection
            let image_view_create_info = ImageViewCreateInfo::from_image(&image);
            let image_view = ImageView::new(image.clone(), image_view_create_info)
                .expect(format!("Failed to create image view for {}", texture.filepath).as_str());

            let sampler_create_info = SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                lod: 0.0..=LOD_CLAMP_NONE,
                ..Default::default()
            };

            let sampler = Sampler::new(device.clone(), sampler_create_info).unwrap();

            let texture_data = TextureData { image_view, sampler };
            new_texture_data.insert(texture.filepath.clone(), texture_data);

            // Register GPU copy command
            let copy_buffer_to_image_info = match image_data.data
            {
                // Unfortunately need to repeat the buffer creation code for SDR and HDR due
                // to the data types
                PixelData::SDR(data) =>
                {
                    let staging_buffer = match Buffer::from_iter(
                        memory_allocator.clone(),
                        staging_buffer_create_info,
                        staging_buffer_alloc_create_info,
                        data,
                    )
                    {
                        Ok(v) => v,
                        Err(e) =>
                        {
                            let error_msg = format!(
                                "allocate_texture_samplers :: An error occurred while \
                                creating buffer for SDR image {} :: {}. \
                                Skipping image upload.",
                                texture.filepath, e
                            );

                            Self::log_error(error_msg.as_str());
                            continue;
                        }
                    };

                    CopyBufferToImageInfo::buffer_image(staging_buffer, image.clone())
                }

                PixelData::HDR(data) =>
                {
                    let staging_buffer = match Buffer::from_iter(
                        memory_allocator.clone(),
                        staging_buffer_create_info,
                        staging_buffer_alloc_create_info,
                        data,
                    )
                    {
                        Ok(v) => v,
                        Err(e) =>
                        {
                            let error_msg = format!(
                                "allocate_texture_samplers :: An error occurred while \
                                creating buffer for HDR image {} :: {}. \
                                Skipping image upload.",
                                texture.filepath, e
                            );

                            Self::log_error(error_msg.as_str());
                            continue;
                        }
                    };

                    CopyBufferToImageInfo::buffer_image(staging_buffer, image.clone())
                }
            };

            command_buffer_builder.copy_buffer_to_image(copy_buffer_to_image_info).expect(
                format!(
                    "Failed to register copy buffer to image command for: {}",
                    texture.filepath
                )
                .as_str(),
            );

            // Generate mipmaps
            Self::register_mipmap_generation_commands(&mut command_buffer_builder, &image);
        }

        // Execute command buffer and wait until it is done
        command_buffer_builder
            .build()
            .unwrap()
            .execute(graphics_queue.clone())
            .expect("Failed to upload image to GPU")
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None) // Block until image is copied
            .ok();

        new_texture_data
    }

    /// Register equirectangular map to cubemap conversion compute commands
    ///
    /// If `process_all_mipmaps` is set to `true`, compute shader will be dispatched for each
    /// mipmap level in the equirectangular texture (outputted cubemap will contain the same number
    /// of mipmaps). If it is set to `false`, cubemap's [Image]'s number of mipmaps will be
    /// calculated from its dimensions, but compute shader will only be dispatched for the first one
    ///
    /// Output [TexturedData] is returned\
    /// `texture_path` is only used for logging errors
    fn register_cubemap_conversion_commands(
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        cubemap_conversion_pipeline_data: &CubemapComputePipelineData,
        compute_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        equirectangular_map_texture_data: &TextureData,
        process_all_mipmaps: bool,
        texture_path: &str,
    ) -> TextureData
    {
        // Calculate image extents and mipmaps
        // Equirectangular image width is usually 4 times the face size
        let face_size =
            (equirectangular_map_texture_data.image_view.image().extent()[0] / 4).max(1);

        let mipmap_num = {
            let max_mipmaps = if face_size < 2 { 1 } else { face_size.ilog2() };

            if process_all_mipmaps
            {
                equirectangular_map_texture_data.image_view.image().mip_levels().min(max_mipmaps)
            }
            else
            {
                max_mipmaps
            }
        };

        // Using STORAGE usage since it is going to be generated in the compute shader
        let mut image_usage = ImageUsage::SAMPLED | ImageUsage::STORAGE;

        if mipmap_num > 1 && !process_all_mipmaps
        {
            // Only set as transfer source and destination when generating mipmaps via graphics blit
            image_usage = image_usage | ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST;
        }

        // Cube maps are handled as an image with 6 layers (cube faces),
        // so the image extent should only be the size for one face
        let image_create_info = ImageCreateInfo {
            flags: ImageCreateFlags::CUBE_COMPATIBLE,
            image_type: ImageType::Dim2d,
            format: equirectangular_map_texture_data.image_view.format(),
            extent: [face_size, face_size, 1],
            array_layers: 6, // 6 faces
            mip_levels: mipmap_num,
            usage: image_usage,
            ..Default::default()
        };

        let image = Image::new(
            memory_allocator.clone(),
            image_create_info,
            AllocationCreateInfo::default(),
        )
        .expect(format!("Failed to create target cubemap image for {texture_path}").as_str());

        let num_of_dispatches = if process_all_mipmaps { mipmap_num } else { 1 };
        for i in 0..num_of_dispatches
        {
            // Create image view for compute shader
            // Cubemaps are an array of 6 images (one per cube face), so it needs to be Dim2dArray
            // (processing each face separately)
            let image_view_create_info = ImageViewCreateInfo {
                view_type: ImageViewType::Dim2dArray,
                subresource_range: ImageSubresourceRange {
                    mip_levels: i..i + 1, // Mipmap to process
                    ..image.subresource_range()
                },
                ..ImageViewCreateInfo::from_image(&image)
            };

            // Create descriptor sets
            let input_image_view = {
                let input_image = equirectangular_map_texture_data.image_view.image();

                if process_all_mipmaps
                {
                    let input_image_view_create_info = ImageViewCreateInfo {
                        subresource_range: ImageSubresourceRange {
                            mip_levels: i..i + 1,
                            ..input_image.subresource_range()
                        },
                        ..ImageViewCreateInfo::from_image(input_image)
                    };

                    ImageView::new(input_image.clone(), input_image_view_create_info)
                        .expect(format!("Failed to create input image view for {texture_path}'s mipmap level {i}").as_str())
                }
                else
                {
                    equirectangular_map_texture_data.image_view.clone()
                }
            };

            let input_write_descriptor_set = WriteDescriptorSet::image_view_sampler(
                0,
                input_image_view.clone(),
                equirectangular_map_texture_data.sampler.clone(),
            );

            let output_image_view = ImageView::new(image.clone(), image_view_create_info)
                .expect(format!("Failed to create target image view for {texture_path}").as_str());

            let output_write_descriptor_set =
                WriteDescriptorSet::image_view(1, output_image_view.clone());

            let descriptor_set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                cubemap_conversion_pipeline_data.pipeline.layout().set_layouts()[0].clone(),
                [input_write_descriptor_set, output_write_descriptor_set],
                [],
            )
            .expect(
                format!("Failed to create cubemap conversion's descriptor set for {texture_path}")
                    .as_str(),
            );

            // Register compute shader commands
            Self::register_compute_dispatch_command(
                compute_command_buffer_builder,
                &cubemap_conversion_pipeline_data.pipeline,
                descriptor_set,
                [face_size / 16, face_size / 16, 6],
                texture_path,
            );
        }

        // Create actual cubemap image view and store it with current sampler so that it is
        // updated after this function returns
        let cube_image_view_create_info = ImageViewCreateInfo {
            view_type: ImageViewType::Cube,
            ..ImageViewCreateInfo::from_image(&image)
        };

        let cube_image_view = ImageView::new(image.clone(), cube_image_view_create_info)
            .expect(format!("Failed to create cubemap image view for {}", texture_path).as_str());

        TextureData {
            image_view: cube_image_view,
            sampler: equirectangular_map_texture_data.sampler.clone(),
        }
    }

    /// Register compute bind and dispatch commands\
    /// `log_identifier` is only used for log messages
    fn register_compute_dispatch_command(
        compute_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        compute_pipeline: &Arc<ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
        dispatch_groups: [u32; 3],
        log_identifier: &str,
    )
    {
        compute_command_buffer_builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .expect(format!("Failed to bind compute pipeline for {log_identifier}").as_str())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .expect(
                format!("Failed to bind compute descriptor sets for {log_identifier}").as_str(),
            );

        unsafe {
            compute_command_buffer_builder
                .dispatch(dispatch_groups)
                .expect(format!("Failed to dispatch compute shader for {log_identifier}").as_str());
        }
    }

    /// Register commands for processing an environment cubemap in a compute shader\
    ///
    /// If `process_all_mipmaps` is set to `true`, compute shader will be dispatched for each
    /// mipmap level in the equirectangular texture (outputted texture will contain the same number
    /// of mipmaps). If it is set to `false`, output [Image]'s number of mipmaps will be
    /// calculated from its dimensions and [CubemapComputePipelineData], but compute shader will
    /// only be dispatched for the first one
    ///
    /// **NOTE:** If [CubemapComputePipelineData::max_mipmap_num] is higher than 1, this parameter
    /// will be ignored since the shader is supposed to process all mipmaps at once
    ///
    /// Output [TextureData] is returned\
    /// `texture_path` is only used for logging errors
    fn register_environment_map_compute_commands(
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        compute_pipeline_data: &CubemapComputePipelineData,
        compute_command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        cubemap_texture_data: &TextureData,
        dispatch_method: ComputeDispatchMethod,
        texture_path: &str,
    ) -> TextureData
    {
        let source_cubemap_image = cubemap_texture_data.image_view.image();
        let dispatch_once_per_mipmap = match dispatch_method
        {
            ComputeDispatchMethod::OncePerMipmap => true,
            ComputeDispatchMethod::OncePerMipmapAlwaysUseFirstMipmapAsInput => true,
            _ => false,
        };

        // Equirectangular image width is usually 4 times the face size. Height is usually half of
        // the width
        let source_cubemap_image_size =
            [source_cubemap_image.extent()[0], source_cubemap_image.extent()[1]];

        let width = source_cubemap_image_size[0].max(1) * 4;
        let height = width / 2;
        let extent = [width, height, 1];

        let mip_levels = {
            if dispatch_once_per_mipmap && compute_pipeline_data.max_mipmap_num == 1
            {
                cubemap_texture_data.image_view.image().mip_levels()
            }
            else
            {
                if width <= 4 || height < 2
                {
                    1
                }
                else
                {
                    // Final image is going to be a cubemap, so calculate the maximum mipmap level
                    // based on the cubemap dimensions, not the equirectangular dimensions
                    let larger_dimension = width.max(height);
                    let dimensions_max_mipmap_num = (larger_dimension / 4).ilog2();
                    compute_pipeline_data.max_mipmap_num.min(dimensions_max_mipmap_num)
                }
            }
        };

        // Output image is equirectangular, not a cubemap. Cubemap conversion will be done later
        // via the cubemap conversion compute shader
        let output_image_create_info = ImageCreateInfo {
            image_type: source_cubemap_image.image_type(),
            format: source_cubemap_image.format(),
            extent,
            mip_levels,
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        };

        let output_image = Image::new(
            memory_allocator.clone(),
            output_image_create_info,
            AllocationCreateInfo::default(),
        )
        .expect(
            format!("Failed to create output environment map image for {texture_path}").as_str(),
        );

        let output_image_view_create_info = ImageViewCreateInfo::from_image(&output_image);
        let output_image_view = ImageView::new(output_image.clone(), output_image_view_create_info)
            .expect(
                format!("Failed to create output environment map image view for {texture_path}")
                    .as_str(),
            );

        // Dispatch differs depending on whether all mipmaps are being processed, outputted or not
        if dispatch_once_per_mipmap && compute_pipeline_data.max_mipmap_num == 1
        {
            for i in 0..mip_levels
            {
                let input_mipmap_image_view = {
                    if dispatch_method
                        == ComputeDispatchMethod::OncePerMipmapAlwaysUseFirstMipmapAsInput
                    {
                        // If only first mipmap is used as input, use input image view as is
                        cubemap_texture_data.image_view.clone()
                    }
                    else
                    {
                        // When using each mipmap as each input, need to adjust the resource range
                        // each time
                        let input_image = cubemap_texture_data.image_view.image();

                        let input_mipmap_image_view_create_info = ImageViewCreateInfo {
                            view_type: ImageViewType::Cube,
                            subresource_range: ImageSubresourceRange {
                                mip_levels: i..i + 1,
                                ..input_image.subresource_range()
                            },
                            ..ImageViewCreateInfo::from_image(input_image)
                        };

                        ImageView::new(
                            input_image.clone(),
                            input_mipmap_image_view_create_info,
                        ).expect(
                            format!("Failed to create input environment map image view for {texture_path} : mipmap {i}")
                                .as_str(),
                        )
                    }
                };

                let output_mipmap_image_view_create_info = ImageViewCreateInfo {
                    subresource_range: ImageSubresourceRange {
                        mip_levels: i..i + 1,
                        ..output_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::from_image(&output_image)
                };

                let output_mipmap_image_view = ImageView::new(
                    output_image.clone(),
                    output_mipmap_image_view_create_info,
                ).expect(
                    format!("Failed to create output environment map image view for {texture_path} : mipmap {i}")
                        .as_str(),
                );

                // Create descriptor sets
                let input_write_descriptor_set = WriteDescriptorSet::image_view_sampler(
                    0,
                    input_mipmap_image_view,
                    cubemap_texture_data.sampler.clone(),
                );

                let output_write_descriptor_set =
                    WriteDescriptorSet::image_view(1, output_mipmap_image_view);

                // Create descriptor set and register dispatch command
                let descriptor_set = DescriptorSet::new(
                    descriptor_set_allocator.clone(),
                    compute_pipeline_data.pipeline.layout().set_layouts()[0].clone(),
                    [input_write_descriptor_set.clone(), output_write_descriptor_set],
                    [],
                )
                .expect(
                    format!(
                        "Failed to create output environment map descriptor set for {texture_path}"
                    )
                    .as_str(),
                );

                let mipmap_extent = [extent[0] / u32::pow(2, i), extent[1] / u32::pow(2, i)];
                let dispatch_groups = [mipmap_extent[0] / 16, mipmap_extent[1] / 16];

                // Register compute dispatch commands
                if compute_pipeline_data.uses_metadata_constants
                {
                    let environment_map_compute_constants = EnvironmentMapComputeConstants {
                        input_image_size: source_cubemap_image_size,
                        current_mipmap: i,
                        total_mipmap_num: mip_levels,
                    };

                    compute_command_buffer_builder
                        .push_constants(
                            compute_pipeline_data.pipeline.layout().clone(),
                            0,
                            environment_map_compute_constants,
                        )
                        .expect(
                            format!("Failed to bind compute descriptor sets for {texture_path}")
                                .as_str(),
                        );
                }

                Self::register_compute_dispatch_command(
                    compute_command_buffer_builder,
                    &compute_pipeline_data.pipeline,
                    descriptor_set,
                    [dispatch_groups[0].max(1), dispatch_groups[1].max(1), 1],
                    texture_path,
                );
            }
        }
        else
        {
            // When not dispatching once per mipmap, just need to set the output descriptor set
            // to the number of mipmaps that will be outputted from the shader

            // Create descriptor sets
            let input_write_descriptor_set = WriteDescriptorSet::image_view_sampler(
                0,
                cubemap_texture_data.image_view.clone(),
                cubemap_texture_data.sampler.clone(),
            );

            let output_write_descriptor_set = {
                if compute_pipeline_data.max_mipmap_num == 1
                {
                    WriteDescriptorSet::image_view(1, output_image_view.clone())
                }
                else
                {
                    // If outputting multiple mipmaps, need to include each mipmap in the descriptor set
                    let mut output_image_views = Vec::new();

                    // Need to include the maximum number of descriptors in the compute shader
                    // even if it is dummy data
                    for i in 0..compute_pipeline_data.max_mipmap_num
                    {
                        // Only include up to the actual maximum mip level
                        let mipmap_index = i.min(mip_levels - 1);

                        let output_mipmap_image_view_create_info = ImageViewCreateInfo {
                            subresource_range: ImageSubresourceRange {
                                mip_levels: mipmap_index..mipmap_index + 1,
                                ..output_image.subresource_range()
                            },
                            ..ImageViewCreateInfo::from_image(&output_image)
                        };

                        let output_mipmap_image_view = ImageView::new(output_image.clone(), output_mipmap_image_view_create_info).expect(
                            format!("Failed to create output environment map image view for {texture_path}, mipmap {i}")
                                .as_str(),
                        );

                        output_image_views.push(output_mipmap_image_view);
                    }

                    WriteDescriptorSet::image_view_array(1, 0, output_image_views)
                }
            };

            // Create descriptor set and register dispatch command
            let descriptor_set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                compute_pipeline_data.pipeline.layout().set_layouts()[0].clone(),
                [input_write_descriptor_set.clone(), output_write_descriptor_set],
                [],
            )
            .expect(
                format!(
                    "Failed to create output environment map descriptor set for {texture_path}"
                )
                .as_str(),
            );

            // Register compute dispatch commands
            if compute_pipeline_data.uses_metadata_constants
            {
                let environment_map_compute_constants = EnvironmentMapComputeConstants {
                    input_image_size: source_cubemap_image_size,
                    current_mipmap: 0,
                    total_mipmap_num: mip_levels,
                };

                compute_command_buffer_builder
                    .push_constants(
                        compute_pipeline_data.pipeline.layout().clone(),
                        0,
                        environment_map_compute_constants,
                    )
                    .expect(
                        format!("Failed to bind compute descriptor sets for {texture_path}")
                            .as_str(),
                    );
            }

            Self::register_compute_dispatch_command(
                compute_command_buffer_builder,
                &compute_pipeline_data.pipeline,
                descriptor_set,
                [extent[0] / 16, extent[1] / 16, mip_levels],
                texture_path,
            );
        }

        TextureData {
            image_view: output_image_view.clone(),
            sampler: cubemap_texture_data.sampler.clone(),
        }
    }

    /// Register mipmap generation blit commands for an image\
    /// `command_buffer_builder` has to be from a graphics [Queue]
    fn register_mipmap_generation_commands(
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image: &Arc<Image>,
    )
    {
        let image_width = image.extent()[0];
        let image_height = image.extent()[1];

        // For each mipmap, downscale previous mipmap by half
        let mut next_mipmap_width = image_width;
        let mut next_mipmap_height = image_height;

        if image.mip_levels() > 1
        {
            for i in 1..image.mip_levels()
            {
                let src_subresource_layer =
                    ImageSubresourceLayers { mip_level: i - 1, ..image.subresource_layers() };

                let dst_subresource_layer =
                    ImageSubresourceLayers { mip_level: i, ..image.subresource_layers() };

                let previous_width = next_mipmap_width;
                let previous_height = next_mipmap_height;

                next_mipmap_width = cmp::max(1, next_mipmap_width / 2);
                next_mipmap_height = cmp::max(1, next_mipmap_height / 2);

                let blit_region = ImageBlit {
                    src_subresource: src_subresource_layer,
                    src_offsets: [[0; 3], [previous_width, previous_height, 1]],
                    dst_subresource: dst_subresource_layer,
                    dst_offsets: [[0; 3], [next_mipmap_width, next_mipmap_height, 1]],
                    ..Default::default()
                };

                let blit_image_info = BlitImageInfo {
                    regions: Vec::from([blit_region]).into(),
                    filter: Filter::Linear,
                    ..BlitImageInfo::images(image.clone(), image.clone())
                };

                command_buffer_builder
                    .blit_image(blit_image_info)
                    .expect(format!("Failed to generate mipmap {i}").as_str());
            }
        }
    }

    /// BRDF integration map generation
    fn generate_brdf_integration_map(
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        compute_queue: Arc<Queue>,
        size: [u32; 2],
    ) -> TextureData
    {
        let device = compute_queue.device().clone();

        // Create output image and image view for BRDF integration map
        let output_image_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16_SFLOAT,
            extent: [size[0], size[1], 1],
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        };

        let output_image = Image::new(
            memory_allocator.clone(),
            output_image_create_info,
            AllocationCreateInfo::default(),
        )
        .expect("Failed to create image for BRDF integration map");

        let output_image_view_create_info = ImageViewCreateInfo::from_image(&output_image);
        let output_image_view = ImageView::new(output_image, output_image_view_create_info)
            .expect("Failed to create image view for BRDF integration map");

        // Create BRDF integration compute pipeline (this function is expected to be only called
        // once in the entire application, so it should be fine to create the pipeline here since
        // it is not going to be reused later)
        let shader = Self::load_shader(device.clone(), BRDF_INTEGRATION_MAP_SHADER_PATH);
        let shader_stage_create_info = PipelineShaderStageCreateInfo::new(shader);

        let output_descriptor_set_layout_binding = DescriptorSetLayoutBinding {
            stages: ShaderStages::COMPUTE,
            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
        };

        let output_descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings: BTreeMap::from([(0, output_descriptor_set_layout_binding)]),
            ..DescriptorSetLayoutCreateInfo::default()
        };

        let output_descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), output_descriptor_set_layout_create_info)
                .expect("Failed to create output descriptor set layout for BRDF integration");

        let pipeline_layout_create_info = PipelineLayoutCreateInfo {
            set_layouts: vec![output_descriptor_set_layout],
            ..PipelineLayoutCreateInfo::default()
        };

        let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_create_info)
            .expect("Failed to create compute pipeline layout for BRDF integration");

        let pipeline_create_info = ComputePipelineCreateInfo {
            ..ComputePipelineCreateInfo::stage_layout(shader_stage_create_info, pipeline_layout)
        };

        let pipeline = ComputePipeline::new(device.clone(), None, pipeline_create_info)
            .expect("Failed to create cubemap compute pipeline");

        // Create command buffer builder, descriptor set, and register compute shader commands
        let mut compute_command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create compute command buffer for BRDF integration map generation");

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view(0, output_image_view.clone())],
            [],
        )
        .expect("Failed to create descriptor set for BRDF integration");

        Self::register_compute_dispatch_command(
            &mut compute_command_buffer_builder,
            &pipeline,
            descriptor_set,
            [size[0] / 16, size[1] / 16, 1],
            "BRDF integration map",
        );

        // Execute compute command buffer and wait until it is done
        compute_command_buffer_builder
            .build()
            .expect("Failed to build BRDF integration compute command buffer builder")
            .execute(compute_queue.clone())
            .expect("Failed to execute BRDF integration compute commands")
            .then_signal_fence_and_flush()
            .expect("Failed to register fence signal for BRDF integration compute execute")
            .wait(None) // Block until shader is finished
            .expect("Failed to wait for BRDF integration compute dispatch");

        // Create sampler and return texture data
        let sampler_create_info = SamplerCreateInfo {
            min_filter: Filter::Linear,
            mag_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::ClampToEdge; 3],
            ..Default::default()
        };

        let sampler = Sampler::new(device.clone(), sampler_create_info)
            .expect("Failed to create sampler for BRDF integration map");

        TextureData { image_view: output_image_view, sampler }
    }

    /// Allocate vertex and index buffers on the device using given [Mesh] data
    fn allocate_vertex_data_buffers(
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        mesh: &Mesh,
    ) -> (Subbuffer<[VertexData]>, Subbuffer<[u32]>)
    {
        // Get vertex data from mesh
        let mut vertices: Vec<VertexData> = Vec::with_capacity(mesh.vertices.len());
        for (i, vertex) in mesh.vertices.iter().enumerate()
        {
            let data = VertexData {
                position: vertex.to_array(),
                normal: mesh.normals[i].to_array(),
                tangent: mesh.tangents[i].to_array(),
                uv0: mesh.uv_sets[0][i].to_array(),
                uv1: mesh.uv_sets[1][i].to_array(),
                color: mesh.colors[i].to_array(),
            };

            vertices.push(data);
        }

        let num_of_vertices = vertices.len();

        // Since data won't change frequently, using device local buffers is more performant
        // However, need a staging buffer first to store the data and then send it to the GPU
        let staging_buffer_create_info =
            BufferCreateInfo { usage: BufferUsage::TRANSFER_SRC, ..Default::default() };

        let staging_allocation_create_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let device_local_allocation_create_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        };

        // Create vertex buffers (staging and device local)
        let vertex_staging_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            staging_buffer_create_info.clone(),
            staging_allocation_create_info.clone(),
            vertices,
        )
        .unwrap();

        let vertex_buffer_create_info = BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let vertex_buffer = Buffer::new_slice::<VertexData>(
            memory_allocator.clone(),
            vertex_buffer_create_info,
            device_local_allocation_create_info.clone(),
            num_of_vertices as DeviceSize,
        )
        .unwrap();

        // Create index buffer
        let index_staging_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            staging_buffer_create_info,
            staging_allocation_create_info,
            mesh.indices.clone(),
        )
        .unwrap();

        let index_buffer_create_info = BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        };

        let index_buffer = Buffer::new_slice::<u32>(
            memory_allocator,
            index_buffer_create_info,
            device_local_allocation_create_info,
            mesh.indices.len() as DeviceSize,
        )
        .unwrap();

        // Copy buffers to GPU using a temporary command buffer
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let vertex_copy_buffer_info =
            CopyBufferInfo::buffers(vertex_staging_buffer, vertex_buffer.clone());

        let index_copy_buffer_info =
            CopyBufferInfo::buffers(index_staging_buffer, index_buffer.clone());

        command_buffer_builder
            .copy_buffer(vertex_copy_buffer_info)
            .unwrap()
            .copy_buffer(index_copy_buffer_info)
            .unwrap();

        command_buffer_builder
            .build()
            .unwrap()
            .execute(queue)
            .expect("Failed to execute command buffer")
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap(); // Block until copy is executed

        (vertex_buffer, index_buffer)
    }

    /// Allocate the given struct data using a [MemoryAllocator]
    fn allocate_data_buffer<T: BufferContents>(
        memory_allocator: Arc<dyn MemoryAllocator>,
        data: T,
    ) -> Subbuffer<T>
    {
        let buffer_create_info =
            BufferCreateInfo { usage: BufferUsage::UNIFORM_BUFFER, ..Default::default() };

        let allocation_create_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        Buffer::from_data(memory_allocator, buffer_create_info, allocation_create_info, data)
            .unwrap()
    }

    /// Create and initialize the cubemap processing compute pipelines for the given shaders\
    ///
    /// Each element in `pipeline_descriptions` is a tuple with the following:
    /// - Shader path
    /// - Number of mipmaps that will be output from that shader
    /// - Whether the shader needs the image and mipmap metadata push constants or not
    ///
    /// # Shader Requirements
    /// Shaders are expected to have a single input and output. Input is a combined image sampler;
    /// output is a storage image
    fn create_cubemap_processing_pipelines(
        device: &Arc<Device>,
        pipeline_descriptions: Vec<(impl AsRef<Path>, u32, bool)>,
    ) -> Vec<CubemapComputePipelineData>
    {
        let mut pipelines = Vec::new();

        for (shader_path, mipmap_num, uses_metadata_constants) in pipeline_descriptions
        {
            // Load shader and create shader stage create info
            let shader = Self::load_shader(device.clone(), shader_path);
            let shader_stage_create_info = PipelineShaderStageCreateInfo::new(shader);

            // Create descriptor set layout with input and output textures
            let input_descriptor_set_layout_binding = DescriptorSetLayoutBinding {
                descriptor_count: 1,
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
            };

            let output_descriptor_set_layout_binding = DescriptorSetLayoutBinding {
                descriptor_count: mipmap_num,
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
            };

            let descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
                bindings: BTreeMap::from([
                    (0, input_descriptor_set_layout_binding),
                    (1, output_descriptor_set_layout_binding),
                ]),
                ..DescriptorSetLayoutCreateInfo::default()
            };

            let descriptor_set_layout =
                DescriptorSetLayout::new(device.clone(), descriptor_set_layout_create_info)
                    .expect("Failed to create cubemap compute input descriptor set layout");

            let push_constant_range = {
                if uses_metadata_constants
                {
                    let range = PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        offset: 0,
                        size: size_of::<EnvironmentMapComputeConstants>() as u32,
                    };

                    Vec::from([range])
                }
                else
                {
                    Vec::new()
                }
            };

            // Create pipeline
            let pipeline_layout_create_info = PipelineLayoutCreateInfo {
                set_layouts: vec![descriptor_set_layout],
                push_constant_ranges: push_constant_range,
                ..PipelineLayoutCreateInfo::default()
            };

            let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_create_info)
                .expect("Failed to create cubemap compute pipeline layout");

            let pipeline_create_info = ComputePipelineCreateInfo {
                ..ComputePipelineCreateInfo::stage_layout(shader_stage_create_info, pipeline_layout)
            };

            let pipeline = ComputePipeline::new(device.clone(), None, pipeline_create_info)
                .expect("Failed to create cubemap compute pipeline");

            pipelines.push(CubemapComputePipelineData {
                pipeline,
                max_mipmap_num: mipmap_num,
                uses_metadata_constants,
            });
        }

        pipelines
    }

    /// Create and initialize the graphics pipeline that will be used for rendering 3D objects
    fn create_material_graphics_pipeline(
        device: &Arc<Device>,
        swapchain: &Arc<Swapchain>,
        render_pass: &Arc<RenderPass>,
        vert_shader_entry_point: EntryPoint,
        frag_shader_entry_point: EntryPoint,
        texture_count: u32,
        lighting_mode: LightingMode,
    ) -> Arc<GraphicsPipeline>
    {
        let vertex_input_state =
            VertexData::per_vertex().definition(&vert_shader_entry_point).unwrap();

        let scissor = Scissor { offset: [0; 2], extent: swapchain.image_extent() };

        let viewport_state = ViewportState {
            scissors: [scissor].into_iter().collect(),
            ..Default::default() // Not setting viewport since it is resized dynamically
        };

        let rasterization_state = RasterizationState {
            cull_mode: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            ..Default::default()
        };

        let color_blend_state = ColorBlendState {
            attachments: Vec::from([ColorBlendAttachmentState::default()]),
            ..Default::default()
        };

        let depth_stencil_state =
            DepthStencilState { depth: Some(DepthState::simple()), ..Default::default() };

        let subpass = Subpass::from(render_pass.clone(), 0).expect("Failed to create subpass");

        let pipeline_stages = [
            PipelineShaderStageCreateInfo::new(vert_shader_entry_point),
            PipelineShaderStageCreateInfo::new(frag_shader_entry_point),
        ];

        // Declare push constant range for objects' transform matrix,
        // descriptor set layout for view and perspective matrices,
        // and the shared data for fragment shaders
        let push_constant_range = PushConstantRange {
            stages: ShaderStages::VERTEX,
            offset: 0,
            size: size_of::<ObjectTransformData>() as u32,
        };

        let view_proj_descriptor_set_layout_binding = DescriptorSetLayoutBinding {
            descriptor_type: DescriptorType::UniformBuffer,
            descriptor_count: 1,
            stages: ShaderStages::VERTEX,
            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
        };

        let fragment_shader_extra_data_set_layout_binding = DescriptorSetLayoutBinding {
            descriptor_type: DescriptorType::UniformBuffer,
            descriptor_count: 1,
            stages: ShaderStages::FRAGMENT,
            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
        };

        // Create descriptor set layout for view-projection matrices
        let view_proj_descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings: BTreeMap::from([(0, view_proj_descriptor_set_layout_binding)]),
            ..DescriptorSetLayoutCreateInfo::default()
        };

        let view_proj_descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), view_proj_descriptor_set_layout_create_info)
                .expect("Failed to create descriptor set layout for view-projection matrices");

        let fragment_shader_extra_data_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings: BTreeMap::from([(0, fragment_shader_extra_data_set_layout_binding)]),
            ..DescriptorSetLayoutCreateInfo::default()
        };

        let fragment_shader_extra_data_set_layout = DescriptorSetLayout::new(
            device.clone(),
            fragment_shader_extra_data_set_layout_create_info,
        )
        .expect("Failed to create descriptor set layout for fragment shader extra data");

        // Create pipeline layout
        let mut descriptor_set_layouts =
            Vec::from([view_proj_descriptor_set_layout, fragment_shader_extra_data_set_layout]);

        // If using textures, create textures' descriptor set layout and add it to pipeline layout
        if texture_count > 0
        {
            let textures_descriptor_set_layout_binding = DescriptorSetLayoutBinding {
                descriptor_type: DescriptorType::CombinedImageSampler,
                descriptor_count: 1,
                stages: ShaderStages::FRAGMENT,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
            };

            // Create one binding per texture
            let texture_bindings: Vec<_> = (0..texture_count)
                .map(|i| (i, textures_descriptor_set_layout_binding.clone()))
                .collect();

            let textures_descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
                bindings: BTreeMap::from_iter(texture_bindings),
                ..DescriptorSetLayoutCreateInfo::default()
            };

            let textures_descriptor_set_layout = DescriptorSetLayout::new(
                device.clone(),
                textures_descriptor_set_layout_create_info,
            )
            .unwrap();

            descriptor_set_layouts.push(textures_descriptor_set_layout);
        };

        // If using lighting, create environment map data descriptor set layout
        if lighting_mode == LightingMode::Lit || lighting_mode == LightingMode::PBR
        {
            descriptor_set_layouts.push(Self::get_environment_map_data_layout(device.clone()));

            // If PBR, add PBR parameters' descriptor set layout
            if lighting_mode == LightingMode::PBR
            {
                let pbr_params_descriptor_set_layout_binding = DescriptorSetLayoutBinding {
                    descriptor_count: 1,
                    stages: ShaderStages::FRAGMENT,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                };

                let pbr_params_descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
                    bindings: BTreeMap::from([(0, pbr_params_descriptor_set_layout_binding)]),
                    ..DescriptorSetLayoutCreateInfo::default()
                };

                let pbr_params_descriptor_set_layout = DescriptorSetLayout::new(
                    device.clone(),
                    pbr_params_descriptor_set_layout_create_info,
                )
                .expect("Failed to create PBR parameters descriptor set layout");

                descriptor_set_layouts.push(pbr_params_descriptor_set_layout);
            }
        }

        let pipeline_layout_create_info = PipelineLayoutCreateInfo {
            push_constant_ranges: Vec::from([push_constant_range]),
            set_layouts: descriptor_set_layouts,
            ..PipelineLayoutCreateInfo::default()
        };

        let pipeline_layout =
            PipelineLayout::new(device.clone(), pipeline_layout_create_info).unwrap();

        let graphics_pipeline_create_info = GraphicsPipelineCreateInfo {
            stages: pipeline_stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(viewport_state),
            rasterization_state: Some(rasterization_state),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(color_blend_state),
            depth_stencil_state: Some(depth_stencil_state),
            subpass: Some(subpass.into()),
            // Need to resize viewport if window is resized, so need to set as dynamic
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        };

        GraphicsPipeline::new(device.clone(), None, graphics_pipeline_create_info)
            .expect("Failed to create material graphics pipeline")
    }

    /// Create and initialize the graphics pipeline that will be used for post-processing
    fn create_post_process_graphics_pipeline(
        device: &Arc<Device>,
        swapchain: &Arc<Swapchain>,
        render_pass: &Arc<RenderPass>,
        vert_shader_entry_point: EntryPoint,
        frag_shader_entry_point: EntryPoint,
        subpass_index: u32,
    ) -> Arc<GraphicsPipeline>
    {
        let scissor = Scissor { offset: [0; 2], extent: swapchain.image_extent() };

        let viewport_state = ViewportState {
            scissors: [scissor].into_iter().collect(),
            ..Default::default() // Not setting viewport since it is resized dynamically
        };

        let rasterization_state = RasterizationState {
            cull_mode: CullMode::Front, // To match hard-coded vertices in vertex shader
            front_face: FrontFace::CounterClockwise,
            ..Default::default()
        };

        let color_blend_state = ColorBlendState {
            attachments: Vec::from([ColorBlendAttachmentState::default()]),
            ..Default::default()
        };

        let subpass = Subpass::from(render_pass.clone(), subpass_index).expect(
            format!("Failed to create subpass for subpass index {}", subpass_index).as_str(),
        );

        let pipeline_stages = [
            PipelineShaderStageCreateInfo::new(vert_shader_entry_point),
            PipelineShaderStageCreateInfo::new(frag_shader_entry_point),
        ];

        // Create descriptor set layout for post process attachments (color)
        let descriptor_set_layout_binding = DescriptorSetLayoutBinding {
            descriptor_type: DescriptorType::InputAttachment,
            descriptor_count: 1,
            stages: ShaderStages::FRAGMENT,
            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::InputAttachment)
        };

        let descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings: BTreeMap::from([(0, descriptor_set_layout_binding.clone())]),
            ..DescriptorSetLayoutCreateInfo::default()
        };

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            descriptor_set_layout_create_info,
        )
        .expect(
            format!("Failed to create descriptor set layout for subpass index {}", subpass_index)
                .as_str(),
        );

        // Create push constant ranges (post process extra shader data)
        let push_constant_range = PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            offset: 0,
            size: size_of::<PostProcessShaderData>() as u32,
        };

        let pipeline_layout_create_info = PipelineLayoutCreateInfo {
            push_constant_ranges: Vec::from([push_constant_range]),
            set_layouts: Vec::from([descriptor_set_layout]),
            ..PipelineLayoutCreateInfo::default()
        };

        let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_create_info)
            .expect(
                format!("Failed to create pipeline layout for subpass index {}", subpass_index)
                    .as_str(),
            );

        let graphics_pipeline_create_info = GraphicsPipelineCreateInfo {
            stages: pipeline_stages.into_iter().collect(),
            // Using empty vertex input state since the vertices hard-coded in the shader
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(viewport_state),
            rasterization_state: Some(rasterization_state),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(color_blend_state),
            subpass: Some(subpass.into()),
            // Need to resize viewport if window is resized, so need to set as dynamic
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        };

        GraphicsPipeline::new(device.clone(), None, graphics_pipeline_create_info).expect(
            format!(
                "Failed to create post processing graphics pipeline for subpass index {}",
                subpass_index
            )
            .as_str(),
        )
    }

    /// Create post process render targets and then write them to a descriptor set
    fn create_post_process_attachments(
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
        images_extent: [u32; 3],
        is_hdr: bool,
    ) -> (Vec<Arc<ImageView>>, Arc<DescriptorSet>)
    {
        let format = if is_hdr { HDR_FORMAT } else { SDR_FORMAT };

        let image_create_info = ImageCreateInfo {
            format,
            extent: images_extent,
            usage: ImageUsage::INPUT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
            initial_layout: ImageLayout::Undefined,
            ..Default::default()
        };

        let image = Image::new(
            memory_allocator.clone(),
            image_create_info,
            AllocationCreateInfo::default(),
        )
        .expect("Failed to create input attachment for post process");

        let image_view_create_info = ImageViewCreateInfo::from_image(&image);
        let image_view = ImageView::new(image, image_view_create_info)
            .expect("Failed to create input attachment image view for post process");

        let write_descriptor_set = WriteDescriptorSet::image_view(0, image_view.clone());

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [write_descriptor_set],
            [],
        )
        .expect("Failed to create descriptor set for post process");

        (vec![image_view], descriptor_set)
    }

    /// Create a framebuffer per swapchain image and make them compatible with the given render pass
    /// `subpass_attachments` is a vector of each subpass' attachments per swapchain image
    /// (Each element of `subpass_attachments` has to have the same length as `swapchain_images`)
    ///
    /// # Panics
    /// Will panic if any of the subpasses' set of attachments' length do not match the
    /// `swapchain_images` length
    fn create_framebuffers(
        memory_allocator: Arc<dyn MemoryAllocator>,
        swapchain_images: &Vec<Arc<Image>>,
        render_pass: &Arc<RenderPass>,
        subpasses_attachments: &Vec<Vec<SubpassAttachments>>,
        depth_stencil_format: Format,
    ) -> Vec<Arc<Framebuffer>>
    {
        let mut framebuffers = Vec::new();

        // Create a framebuffer per swapchain image
        for (i, image) in swapchain_images.iter().enumerate()
        {
            // Create image view from each frame buffer image
            let image_view = ImageView::new_default(image.clone()).unwrap();

            // Create depth-stencil image per frame buffer
            let depth_stencil_image_create_info = ImageCreateInfo {
                format: depth_stencil_format,
                extent: image.extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                initial_layout: ImageLayout::Undefined,
                ..Default::default()
            };

            let depth_stencil_allocation_info = AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            };

            let depth_stencil_image = Image::new(
                memory_allocator.clone(),
                depth_stencil_image_create_info,
                depth_stencil_allocation_info,
            )
            .expect("Failed to create depth stencil image");

            let depth_stencil_image_view =
                ImageView::new_default(depth_stencil_image.clone()).unwrap();

            // Gather attachments
            let mut framebuffer_attachments = Vec::from([image_view, depth_stencil_image_view]);
            for subpass_swapchain_copies in subpasses_attachments
            {
                // Check that each subpass' attachments have one copy per swapchain image
                if subpass_swapchain_copies.len() != swapchain_images.len()
                {
                    panic!(
                        "Subpass' attachments length does not match the number of swapchain images"
                    );
                }

                for attachments in subpass_swapchain_copies.get(i)
                {
                    framebuffer_attachments.extend(attachments.clone());
                }
            }

            // Create framebuffer
            let framebuffer_create_info = FramebufferCreateInfo {
                attachments: framebuffer_attachments,
                ..Default::default()
            };

            let framebuffer = Framebuffer::new(render_pass.clone(), framebuffer_create_info)
                .expect("Failed to create framebuffer");

            framebuffers.push(framebuffer);
        }

        framebuffers
    }

    /// Create the textures' descriptors set for a given graphics pipeline using the
    /// given texture assignments and texture data
    fn create_textures_descriptor_set(
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        graphics_pipeline: Arc<GraphicsPipeline>,
        material_name: &str,
        texture_assignments: &HashMap<TextureUsage, String>,
        texture_data: &HashMap<String, TextureData>,
    ) -> Arc<DescriptorSet>
    {
        let mut texture_bindings = Vec::new();

        for (texture_usage, filepath) in texture_assignments.iter()
        {
            let texture_data = texture_data.get(filepath).unwrap();

            let write_descriptor_set = WriteDescriptorSet::image_view_sampler(
                texture_usage.clone() as u32,
                texture_data.image_view.clone(),
                texture_data.sampler.clone(),
            );

            texture_bindings.push(write_descriptor_set);
        }

        let descriptor_set_layout = graphics_pipeline
            .layout()
            .set_layouts()
            .get(TEXTURES_DESCRIPTOR_SET_LAYOUT_INDEX)
            .unwrap()
            .clone();

        DescriptorSet::new(descriptor_set_allocator, descriptor_set_layout, texture_bindings, [])
            .expect(
                format!("Failed to create textures' descriptor set for {material_name}").as_str(),
            )
    }

    /// Load shader from file or retrieve entry point directly from cache.
    /// Cache will be updated with new shader if loaded.
    ///
    /// # Panics
    /// Will panic if `shader path` does not point to a valid shader file
    fn get_shader_entry_point(
        device: &Arc<Device>,
        shader_path: &str,
        cache: &mut HashMap<String, EntryPoint>,
    ) -> EntryPoint
    {
        if cache.contains_key(shader_path)
        {
            cache.get(shader_path).unwrap().clone()
        }
        else
        {
            let entry_point = Self::load_shader(device.clone(), shader_path);
            cache.insert(String::from(shader_path), entry_point.clone());

            entry_point
        }
    }

    /// Load a shader from the given path
    ///
    /// # Panics
    /// It will panic if the file doesn't exist, or if it is not a valid shader file
    fn load_shader(device: Arc<Device>, shader_path: impl AsRef<Path>) -> EntryPoint
    {
        let filepath = Path::new(env!("CARGO_MANIFEST_DIR")).join(shader_path);
        let filepath_str = filepath.to_str().unwrap();

        #[cfg(debug_assertions)]
        println!("Attempting to load shader file: {filepath_str}");

        let mut bytes = Vec::new();
        let mut file =
            File::open(&filepath).expect(&format!("Failed to open shader file {filepath_str}"));

        file.read_to_end(&mut bytes).expect(&format!("Failed to read shader file {filepath_str}"));

        let shader_words = vulkano::shader::spirv::bytes_to_words(&bytes)
            .expect("Failed to read shader words from file {filepath}");

        let shader_module_create_info = ShaderModuleCreateInfo::new(&shader_words);
        let shader_module =
            unsafe { ShaderModule::new(device.clone(), shader_module_create_info).unwrap() };

        shader_module
            .entry_point("main")
            .expect(&format!("Failed to get entry point for shader file {filepath_str}"))
    }

    /// Find the best supported format by the device for use as a depth buffer
    /// # Panics
    /// Will panic if physical device does not support any depth buffer format
    fn get_best_supported_depth_buffer_format(physical_device: &Arc<PhysicalDevice>) -> Format
    {
        // Tiling methods to check, in order from best to worst
        let tiling_methods = [ImageTiling::Optimal, ImageTiling::Linear];

        // Depth formats to check, in order from best to worst
        let depth_formats = [
            Format::D32_SFLOAT_S8_UINT,
            Format::D24_UNORM_S8_UINT,
            Format::D16_UNORM_S8_UINT,
            Format::D32_SFLOAT,
            Format::D16_UNORM,
        ];

        let mut best_depth_format = None;

        for tiling_method in tiling_methods
        {
            for format in depth_formats
            {
                let format_properties = physical_device.format_properties(format).unwrap();

                let tiling_features = match tiling_method
                {
                    ImageTiling::Optimal => format_properties.optimal_tiling_features,
                    ImageTiling::Linear => format_properties.linear_tiling_features,
                    _ => format_properties.optimal_tiling_features,
                };

                if tiling_features.contains(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
                {
                    best_depth_format = Some(format);
                    break;
                }
            }

            if best_depth_format.is_some()
            {
                break;
            }
        }

        // If no format is found, panic
        if best_depth_format.is_none()
        {
            panic!("No suitable depth buffer format found");
        };

        best_depth_format.unwrap()
    }

    /// Get environment map data (buffer + image and sampler) descriptor set layout
    fn get_environment_map_data_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout>
    {
        // Lighting settings data
        let environment_map_data_binding = DescriptorSetLayoutBinding {
            descriptor_count: 1,
            stages: ShaderStages::FRAGMENT,
            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
        };

        // Generic image sampler binding
        let image_sampler_binding = DescriptorSetLayoutBinding {
            descriptor_count: 1,
            stages: ShaderStages::FRAGMENT,
            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
        };

        let environment_map_layout_info = DescriptorSetLayoutCreateInfo {
            bindings: BTreeMap::from([
                (0, environment_map_data_binding),
                (1, image_sampler_binding.clone()), // Irradiance map
                (2, image_sampler_binding.clone()), // Prefiltered map
                (3, image_sampler_binding),         // BRDF integration map
            ]),
            ..DescriptorSetLayoutCreateInfo::default()
        };

        DescriptorSetLayout::new(device.clone(), environment_map_layout_info)
            .expect("Failed to create environment map descriptor set layout")
    }

    /// Get GPU image format from image data
    fn get_image_format_from_image_data(image_data: &ImageData, texture: &Texture) -> Format
    {
        match image_data.data
        {
            PixelData::SDR(_) => match texture.usage
            {
                TextureUsage::Color => Format::R8G8B8A8_SRGB,
                TextureUsage::OcclusionRoughnessMetallic => Format::R8G8B8A8_UNORM,
                TextureUsage::Normal => Format::R8G8B8A8_UNORM,
            },

            PixelData::HDR(_) => Format::R32G32B32A32_SFLOAT,
        }
    }

    /// Log error with the given message
    fn log_error(message: &str)
    {
        println!("Error :: GfxApi :: {message}");
    }

    /// Utility function for converting device type to a string slice for logging
    #[cfg(debug_assertions)]
    fn get_device_type_name(device_type: &PhysicalDeviceType) -> &str
    {
        match device_type
        {
            PhysicalDeviceType::DiscreteGpu => "Discrete GPU",
            PhysicalDeviceType::IntegratedGpu => "Integrated GPU",
            PhysicalDeviceType::VirtualGpu => "Virtual GPU",
            PhysicalDeviceType::Cpu => "CPU",
            PhysicalDeviceType::Other => "Other",
            _ => "Other",
        }
    }

    /// Debug log the depth stencil format
    #[cfg(debug_assertions)]
    fn log_depth_stencil_format(format: &Format)
    {
        let format_str = match format
        {
            Format::D32_SFLOAT_S8_UINT => "D32_SFLOAT_S8_UINT",
            Format::D24_UNORM_S8_UINT => "D24_UNORM_S8_UINT",
            Format::D16_UNORM_S8_UINT => "D16_UNORM_S8_UINT",
            Format::D32_SFLOAT => "D32_SFLOAT",
            Format::D16_UNORM => "D16_UNORM",
            _ => "Unknown",
        };

        println!("Depth stencil format: {format_str}");
    }

    /// Debug log the required Vulkan instance extensions
    #[cfg(debug_assertions)]
    fn log_required_extensions(required_extensions: &InstanceExtensions)
    {
        println!("Required Extensions:");

        for extension in required_extensions.into_iter()
        {
            println!("{}", extension.0);
        }

        println!("-----------------------");
    }

    // --- Methods ---

    /// Notify graphics API that window has been resized so that it updates its rendering
    /// state to match the new window size
    pub fn on_window_resized(&mut self, new_size: PhysicalSize<u32>)
    {
        self.should_recreate_swapchain = true;
        self.window_size = new_size.into();
    }

    /// Initializes the current environment map using the given [ImageData]\
    /// If an environment map is already loaded it will be replaced with the new one
    pub fn init_environment_map(&mut self, image_data: ImageData)
    {
        let (env_map_texture_data, irradiance_map_texture_data, prefiltered_map_texture_data) =
            Self::allocate_environment_map_samplers(
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.descriptor_set_allocator.clone(),
                self.device.clone(),
                self.graphics_queue.clone(),
                self.compute_queue.clone(),
                &self.cubemap_conversion_pipeline_data,
                &self.irradiance_map_pipeline_data,
                &self.prefiltered_map_pipeline_data,
                image_data,
            );

        // Actual descriptor set will be updated after frame rendering is done
        self.environment_texture_data = EnvironmentTextureData {
            environment_map: env_map_texture_data,
            irradiance_map: irradiance_map_texture_data,
            prefiltered_map: prefiltered_map_texture_data,
            brdf_integration_map: self.environment_texture_data.brdf_integration_map.clone(),
        };

        self.updated_environment_map_texture = true;
    }

    /// Initialize a new mesh for rendering (upload data to GPU)
    /// Instance ID for that mesh and material combination will be returned
    pub fn init_mesh(
        &mut self,
        mesh: &Mesh,
        transform_matrix: [[f32; 4]; 4],
        material_name: &String,
    ) -> u32
    {
        // If mesh is already loaded just add transform to its corresponding material ID
        if self.render_meshes_data.contains_key(&mesh.filepath)
        {
            let object_transform_data = ObjectTransformData { matrix: transform_matrix };
            let render_mesh_data = self.render_meshes_data.get_mut(&mesh.filepath).unwrap();

            // Add transform only if material is already registered
            if let Some(transforms_data) =
                render_mesh_data.transform_data_per_pipeline_id.get_mut(material_name)
            {
                transforms_data.push(object_transform_data);
                (transforms_data.len() - 1) as u32
            }
            else
            {
                let transforms_data = Vec::from([object_transform_data]);

                render_mesh_data.transform_data_per_pipeline_id.insert(
                    material_name.clone(),
                    transforms_data
                );

                0 // Return 0 since it is the first instance using this material
            }
        }
        else
        {
            // Allocate buffers
            let (vertex_buffer, index_buffer) = Self::allocate_vertex_data_buffers(
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.graphics_queue.clone(),
                mesh,
            );


            let transform_data = ObjectTransformData { matrix: transform_matrix };
            let render_mesh_data = RenderMeshData {
                vertex_buffer,
                index_buffer,
                transform_data_per_pipeline_id: HashMap::from(
                    [(material_name.clone(), Vec::from([transform_data]))]
                ),
            };

            self.render_meshes_data.insert(mesh.filepath.clone(), render_mesh_data);

            0 // Return 0 since it is the first instance
        }
    }

    /// Initialize vertex and fragment shaders pair (meant for 3D object materials)
    /// # Panics
    /// Will panic if any of the shader files is not found or corrupted
    pub fn init_material_shaders(
        &mut self,
        vertex_shader_path: &str,
        fragment_shader_path: &str,
        material_name: &String,
        texture_count: u32,
        lighting_mode: LightingMode,
    )
    {
        // Load shaders and initialize pipeline
        let vert_shader_entry_point = Self::get_shader_entry_point(
            &self.device,
            vertex_shader_path,
            &mut self.shader_entry_points,
        );

        let frag_shader_entry_point = Self::get_shader_entry_point(
            &self.device,
            fragment_shader_path,
            &mut self.shader_entry_points,
        );

        let graphics_pipeline = Self::create_material_graphics_pipeline(
            &self.device,
            &self.swapchain,
            &self.render_pass,
            vert_shader_entry_point,
            frag_shader_entry_point,
            texture_count,
            lighting_mode,
        );

        // If PBR, create default PBR parameters and upload them to the GPU
        let pbr_parameters_data = match lighting_mode
        {
            LightingMode::PBR =>
            {
                let pbr_parameters = PbrParametersBufferData::default();
                let pbr_parameters_buffer = Self::allocate_data_buffer(
                    self.memory_allocator.clone(),
                    pbr_parameters.clone(),
                );

                let pbr_parameters_descriptor_set_layout = graphics_pipeline
                    .layout()
                    .set_layouts()
                    .get(PBR_PARAMS_DESCRIPTOR_SET_LAYOUT_INDEX)
                    .expect(
                        format!(
                            "Failed to retrieve PBR parameters' descriptor set layout \
                                        for {material_name}"
                        )
                        .as_str(),
                    )
                    .clone();

                let pbr_parameters_descriptor_set = DescriptorSet::new(
                    self.descriptor_set_allocator.clone(),
                    pbr_parameters_descriptor_set_layout,
                    [WriteDescriptorSet::buffer(0, pbr_parameters_buffer.clone())],
                    [],
                )
                .expect(
                    format!(
                        "Failed to create PBR parameters' descriptor set \
                                             for {material_name}"
                    )
                    .as_str(),
                );

                let pbr_parameters_data = PbrParametersData {
                    pbr_parameters,
                    pbr_parameters_buffer,
                    pbr_parameters_descriptor_set,
                    updated_parameters: false,
                };

                Some(pbr_parameters_data)
            }

            _ => None,
        };

        // Add/update pipeline data (actual textures' data has to be set with init_textures)
        let pipeline_data = MaterialPipelineData {
            vert_shader_path: String::from(vertex_shader_path),
            frag_shader_path: String::from(fragment_shader_path),
            graphics_pipeline,
            lighting_mode,
            pbr_parameters_data,
            texture_assignments: HashMap::new(),
            textures_descriptor_set: None,
        };

        self.material_pipeline_data.insert(material_name.clone(), pipeline_data);
    }

    /// Initialize a new post process subpass
    /// If a post process with the `shader_path` has been already initialized, call will be ignored
    /// # Panics
    /// Will panic if any of the shader files is not found or corrupted
    pub fn init_post_process(&mut self, shader_path: &str, priority: u32)
    {
        // Check if shader has already been registered
        if self.post_processing_pipelines.iter().any(|data| data.shader_path == shader_path)
        {
            let message = format!(
                "A post process with shader path: {} \
                has already been registered. Ignoring call",
                shader_path
            );
            Self::log_error(message.as_str());
            return;
        }

        // Register post process
        let pipeline_data = PostProcessPipelineData {
            shader_path: String::from(shader_path),
            priority,
            shader_data: PostProcessShaderData::default(),

            // Will be created after render pass is recreated in "recreate_graphics_pipelines"
            graphics_pipeline: None,

            // These two will be initialized when recreating the framebuffer
            // during "recreate_swapchain"
            attachments_image_views: Vec::new(),
            attachments_descriptor_sets: Vec::new(),
        };

        self.post_processing_pipelines.push(pipeline_data);
        self.post_processing_pipelines.sort_by_key(|data| data.priority);

        // Need to recreate render pass once all post processes are added in a single frame
        self.should_recreate_render_pass = true;

        // Need to recreate framebuffers since extra attachments will be needed
        self.should_recreate_swapchain = true;
    }

    /// Initialize new textures for rendering (upload data to GPU)
    ///
    /// Textures are assigned to specific materials (the data itself is reused if used on multiple),
    /// so make sure to pass the correct material name for the corresponding material
    pub fn init_textures(
        &mut self,
        mut image_data: Vec<(ImageData, &Texture)>,
        material_name: &String,
    )
    {
        // Check first that the pipeline data is already initialized
        if !self.material_pipeline_data.contains_key(material_name)
        {
            let err_msg = format!(
                "init_textures :: Pipeline data for {material_name} is not initialized.\
                Please initialize shader/pipeline data first with init_shaders"
            );

            Self::log_error(err_msg.as_str());
            return;
        }

        // Allocate textures and assign them to the corresponding pipeline
        let mut new_texture_indices = Vec::new();
        let mut new_cubemap_paths = Vec::new();

        for (i, (_, texture)) in image_data.iter_mut().enumerate()
        {
            // Skip texture allocation if already loaded
            if self.texture_data.contains_key(&texture.filepath)
            {
                #[cfg(debug_assertions)]
                println!(
                    "GfxApi :: init_texture :: Texture {} :: Texture is already initialized. \
                    Skipping initialization.",
                    texture.filepath
                );
            }
            else
            {
                if texture.is_cubemap
                {
                    new_cubemap_paths.push(texture.filepath.as_str());
                }

                new_texture_indices.push(i);
            }

            // Assign textures based on usage
            self.material_pipeline_data
                .get_mut(material_name)
                .unwrap()
                .texture_assignments
                .insert(texture.usage.clone(), texture.filepath.clone());
        }

        // Allocate all new non-cubemap texture samplers
        let mut image_data_to_allocate = Vec::new();
        for index in new_texture_indices
        {
            // Move image data to avoid copying unnecessarily
            let moved_data = mem::take(&mut image_data[index].0);
            image_data_to_allocate.push((moved_data, image_data[index].1));
        }

        if !image_data_to_allocate.is_empty()
        {
            let new_texture_data = Self::allocate_texture_samplers(
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.device.clone(),
                self.graphics_queue.clone(),
                image_data_to_allocate,
            );

            self.texture_data.extend(new_texture_data);
        }

        // Allocate and convert new cubemap texture samplers
        if !new_cubemap_paths.is_empty()
        {
            // Get cubemaps source texture data
            let source_cubemap_data = {
                let mut cubemap_data = Vec::new();
                for path in new_cubemap_paths
                {
                    if let Some(data) = self.texture_data.get(path)
                    {
                        cubemap_data.push((path, data));
                    }
                }

                cubemap_data
            };

            // Equirectangular images need to be converted to an actual cubemaps to be used
            let new_cubemap_data = Self::allocate_cubemap_samplers(
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.descriptor_set_allocator.clone(),
                self.graphics_queue.clone(),
                self.compute_queue.clone(),
                &self.cubemap_conversion_pipeline_data,
                &self.irradiance_map_pipeline_data,
                &self.prefiltered_map_pipeline_data,
                source_cubemap_data,
                false,
            );

            // Overwrite source equirectangular textures with the converted cubemap
            self.texture_data.extend(new_cubemap_data);
        }

        // Update texture descriptor set
        let graphics_pipeline = self
            .material_pipeline_data
            .get(material_name)
            .expect(format!("Failed to get graphics pipeline with name: {material_name}").as_str())
            .graphics_pipeline
            .clone();

        let texture_assignments = &self
            .material_pipeline_data
            .get(material_name)
            .expect(format!("Failed to get pipeline data for {material_name}").as_str())
            .texture_assignments;

        let texture_descriptor_set = Self::create_textures_descriptor_set(
            self.descriptor_set_allocator.clone(),
            graphics_pipeline,
            material_name.as_str(),
            texture_assignments,
            &self.texture_data,
        );

        self.material_pipeline_data.get_mut(material_name).unwrap().textures_descriptor_set =
            Some(texture_descriptor_set);
    }

    /// Record and execute rendering commands once the next frame image is ready
    pub fn render(&mut self)
    {
        // Recreate render pass before rendering if it needs to change
        if self.should_recreate_render_pass
        {
            self.recreate_render_pass();
        }

        // If needed, recreate swapchain and framebuffers before rendering
        if self.should_recreate_swapchain
        {
            self.recreate_swapchain();
        }

        // Get next available image and the future that represents when it will become available
        let (image_index, suboptimal, acquire_future) =
            swapchain::acquire_next_image(self.swapchain.clone(), None)
                .expect("Failed to acquire next image from swapchain");

        // Update swapchain if properties don't match the surface's properties
        if suboptimal
        {
            self.should_recreate_swapchain = true;
        }

        // Allocate command buffer and record render pass
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create command buffer builder");

        // Register clear values for all render pass' attachments
        // (Color + depth + one per post process)
        let render_pass_clear_values = {
            let mut clear_values: Vec<Option<ClearValue>> =
                Vec::with_capacity(2 + self.post_processing_pipelines.len());

            clear_values.push(Some([0.1, 0.1, 0.1, 0.0].into())); // LDR Color
            clear_values.push(Some((1.0, 0).into())); // Depth + stencil

            // Add one HDR clear value per post process
            for _ in 0..self.post_processing_pipelines.len()
            {
                clear_values.push(Some([0.0, 0.0, 0.0, 0.0].into()));
            }

            clear_values
        };

        let render_pass_begin_info = RenderPassBeginInfo {
            clear_values: render_pass_clear_values,
            ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index as usize].clone())
        };

        command_buffer_builder
            .begin_render_pass(render_pass_begin_info, Default::default())
            .expect("An error occurred when beginning the render pass")
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .expect("An error occurred when setting up the command buffer's viewport");

        // Draw each mesh
        for (_, mesh_data) in self.render_meshes_data.iter()
        {
            for (pipeline_id, transforms_data) in mesh_data.transform_data_per_pipeline_id.iter()
            {
                // Get corresponding graphics pipeline data and textures descriptor set
                let pipeline_data = {
                    if !self.material_pipeline_data.contains_key(pipeline_id)
                    {
                        let err_msg = format!(
                            "render :: Pipeline data for [{pipeline_id}] not found.\
                            Using fallback pipeline/shaders",
                        );

                        Self::log_error(err_msg.as_str());

                        self.material_pipeline_data
                            .get(&FALLBACK_MATERIAL_PIPELINE_NAME.to_string())
                            .unwrap()
                    }
                    else
                    {
                        self.material_pipeline_data.get(pipeline_id).unwrap()
                    }
                };

                let mut descriptor_sets = vec![
                    self.view_projection_data_descriptor_set.clone(),
                    self.fragment_shader_extra_data.descriptor_set.clone(),
                ];

                // Add textures' descriptor set if it exists
                if let Some(ref textures_descriptor_set) = pipeline_data.textures_descriptor_set
                {
                    descriptor_sets.push(textures_descriptor_set.clone());
                }

                // Add lighting descriptor set if it uses lighting
                if pipeline_data.lighting_mode == LightingMode::Lit
                    || pipeline_data.lighting_mode == LightingMode::PBR
                {
                    descriptor_sets.push(self.environment_map_descriptor_set.clone());

                    // Add PBR parameters descriptor set if PBR
                    if pipeline_data.lighting_mode == LightingMode::PBR
                    {
                        if let Some(ref pbr_parameter) = pipeline_data.pbr_parameters_data
                        {
                            descriptor_sets
                                .push(pbr_parameter.pbr_parameters_descriptor_set.clone());
                        }
                    }
                }

                let id = pipeline_id;

                // Mesh draw commands
                command_buffer_builder
                    .bind_pipeline_graphics(pipeline_data.graphics_pipeline.clone())
                    .expect(format!("Failed to bind pipeline for {}", id).as_str())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline_data.graphics_pipeline.layout().clone(),
                        0,
                        descriptor_sets,
                    )
                    .expect(format!("Failed to bind descriptor sets for {}", id).as_str())
                    .bind_vertex_buffers(0, mesh_data.vertex_buffer.clone())
                    .expect(format!("Failed to bind vertex buffer for {}", id).as_str())
                    .bind_index_buffer(mesh_data.index_buffer.clone())
                    .expect(format!("Failed to bind index buffer for {}", id).as_str());

                // Draw one mesh per transform
                // TODO: Support instanced drawing
                for (i, transform_data) in transforms_data.iter().enumerate()
                {
                    command_buffer_builder.push_constants(
                        pipeline_data.graphics_pipeline.layout().clone(),
                        0,
                        transform_data.clone(),
                    )
                    .expect(
                        format!("Failed to set push constants for {id} - Instance {i}").as_str()
                    );

                    unsafe
                    {
                        command_buffer_builder
                            .draw_indexed(mesh_data.index_buffer.len() as u32, 1, 0, 0, 0)
                    }
                    .expect(
                        format!("Failed to register draw call for {id} - Instance {i}").as_str()
                    );
                }
            }
        }

        // Subpasses' draw commands (no need to bind vertex and index buffers
        // since the vertices are hard-coded in the vertex shader)

        for (i, post_process_data) in self.post_processing_pipelines.iter().enumerate()
        {
            if let Some(ref post_process_graphics_pipeline) = post_process_data.graphics_pipeline
            {
                command_buffer_builder
                    .next_subpass(SubpassEndInfo::default(), SubpassBeginInfo::default())
                    .expect(
                        format!("Failed to move to post processing subpass index {}", i).as_str(),
                    )
                    .bind_pipeline_graphics(post_process_graphics_pipeline.clone())
                    .expect(
                        format!("Failed to bind post processing pipeline for subpass {}", i)
                            .as_str(),
                    )
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        post_process_graphics_pipeline.layout().clone(),
                        0,
                        Vec::from([post_process_data.attachments_descriptor_sets
                            [image_index as usize]
                            .clone()]),
                    )
                    .expect(
                        format!("Failed to bind post processing descriptor set for subpass {}", i)
                            .as_str(),
                    )
                    .push_constants(
                        post_process_graphics_pipeline.layout().clone(),
                        0,
                        post_process_data.shader_data.clone(),
                    )
                    .expect(format!("Failed to bind push constants for subpass {}", i).as_str());

                unsafe { command_buffer_builder.draw(3, 1, 0, 0) }.expect(
                    format!("Failed to register post processing draw call for subpass {}", i)
                        .as_str(),
                );
            }
            else
            {
                Self::log_error(
                    format!(
                        "Graphics pipeline for post process with shader {} is \"None\" \
                        Skipping subpass.",
                        post_process_data.shader_path,
                    )
                    .as_str(),
                );
            }
        }

        command_buffer_builder
            .end_render_pass(Default::default())
            .expect("Failed to end render pass");

        // Submit command buffer, present when previous frame ends and image becomes available,
        // then signal fence once presentation is finished
        let command_buffer = command_buffer_builder.build().unwrap();

        // Check if we should wait on a previous frame
        if let Some(_) = &self.previous_frame_end
        {
            // Wait until previous frame has finished so that buffers/resources can be updated
            self.previous_frame_end.as_mut().unwrap().wait(None).unwrap();

            // Cleanup finished futures
            self.previous_frame_end.as_mut().unwrap().cleanup_finished();
        }

        // Update necessary buffers before execution
        self.update_view_projection_buffer();
        self.update_fragment_shader_extra_data_buffer();
        self.update_lighting_buffers();
        self.update_pbr_parameters_buffers();

        let swapchain_present_info =
            SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index);

        // Execute command buffer, present, and then signal fence
        let present_fence = sync::now(self.device.clone())
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.graphics_queue.clone(), swapchain_present_info)
            .then_signal_fence_and_flush();

        // Check fence's result once it is finished
        match present_fence.map_err(Validated::unwrap)
        {
            Ok(present_fence) =>
            {
                self.previous_frame_end = Some(present_fence);
            }

            Err(VulkanError::OutOfDate) =>
            {
                self.should_recreate_swapchain = true;
                self.previous_frame_end = None;
            }

            Err(err) =>
            {
                Self::log_error(format!("render :: Failed to flush present fence: {err}").as_str());
                self.previous_frame_end = None;
            }
        }
    }

    /// Update current view-perspective matrices for rendering
    pub fn update_camera(
        &mut self,
        camera_position: [f32; 3],
        view_matrix: [[f32; 4]; 4],
        perspective_matrix: [[f32; 4]; 4],
    )
    {
        // Need to flip Y axis of projection matrix to match Vulkan's viewport coordinates
        // (glam library calculates a perspective matrix meant for OpenGL)
        let mut flipped_perspective_matrix = perspective_matrix;
        flipped_perspective_matrix[1][1] *= -1.0;

        self.view_projection_data =
            ViewProjectionData { view_matrix, perspective_matrix: flipped_perspective_matrix };

        self.fragment_shader_extra_data.data.camera_pos = camera_position;
        self.fragment_shader_extra_data.updated = true;
    }

    /// Update current image-based lighting settings
    pub fn update_lighting(&mut self, exposure: f32, angle: f32)
    {
        if exposure != self.environment_map_data.exposure
            || angle != self.environment_map_data.angle
        {
            self.environment_map_data.exposure = exposure;
            self.environment_map_data.angle = angle;
            self.updated_environment_map_data = true;
        }
    }

    /// Update specific mesh's transform
    ///
    /// Call will be ignored if mesh has not been initialized via [init_mesh](GfxApi::init_mesh)
    /// or if the instance ID is invalid
    pub fn update_mesh_transform(
        &mut self,
        filepath: &String,
        material_name: &String,
        instance_id: u32,
        new_matrix: [[f32; 4]; 4],
    )
    {
        if let Some(render_data) = self.render_meshes_data.get_mut(filepath)
        {
            if let Some(transform_data) =
                render_data.transform_data_per_pipeline_id.get_mut(material_name)
            {
                if instance_id < transform_data.len() as u32
                {
                    transform_data[instance_id as usize].matrix = new_matrix;
                }
                else
                {
                    let err_msg = format!(
                        "update_mesh_transform :: Mesh: {filepath} :: Material: {material_name}\
                        Instance ID {instance_id} is invalid. Ignoring call."
                    );

                    Self::log_error(err_msg.as_str());
                }
            }
            else
            {
                let err_msg = format!(
                    "update_mesh_transform :: Mesh: {filepath} ::\
                    Material: {material_name} is invalid. Ignoring call."
                );

                Self::log_error(err_msg.as_str());
            }
        }
        else
        {
            let err_msg = format!(
                "update_mesh_transform :: Mesh: {filepath} ::\
                Mesh has not been initialized. Ignoring call."
            );

            Self::log_error(err_msg.as_str());
        }
    }
    /// Update PBR parameters for a specific material/pipeline\
    /// Call will be ignored if material does not exist
    pub fn update_pbr_parameters(&mut self, material_name: &String, pbr_parameters: &PbrParameters)
    {
        if !self.material_pipeline_data.contains_key(material_name)
        {
            let err_msg = format!(
                "update_pbr_parameters :: Material: {material_name} ::\
                Material has not been initialized. Ignoring call."
            );

            Self::log_error(err_msg.as_str());

            return;
        }

        let pbr_parameters_data = self
            .material_pipeline_data
            .get_mut(material_name)
            .unwrap()
            .pbr_parameters_data
            .as_mut();

        // Update data if it exists
        // Otherwise log an error since shaders were not initialized as PBR
        match pbr_parameters_data
        {
            Some(data) =>
            {
                data.pbr_parameters.color_tint = pbr_parameters.color_tint;
                data.pbr_parameters.occlusion_strength = pbr_parameters.occlusion_strength;
                data.pbr_parameters.roughness_strength = pbr_parameters.roughness_strength;
                data.pbr_parameters.metallic_strength = pbr_parameters.metallic_strength;
                data.pbr_parameters.normal_strength = pbr_parameters.normal_strength;

                data.updated_parameters = true;
            }

            None =>
            {
                let err_msg = format!(
                    "update_pbr_parameters :: Material: {material_name} ::\
                        Material was not initialized as PBR. Ignoring call."
                );

                Self::log_error(err_msg.as_str());
            }
        }
    }

    /// Update post process parameters
    /// Call will be ignored if post process has not been registered
    pub fn update_post_process_parameters(&mut self, shader_path: &str, strength: f32)
    {
        let post_process_data_option =
            self.post_processing_pipelines.iter_mut().find(|data| data.shader_path == shader_path);

        if let Some(post_process_data) = post_process_data_option
        {
            post_process_data.shader_data.strength = strength;
        }
    }

    /// Update fragment shaders' extra data buffer if needed
    /// # Panics
    /// Will panic if this is called while the GPU is using the uniform buffer\
    /// Need to wait until the previous frame is finished processing before calling this function
    fn update_fragment_shader_extra_data_buffer(&mut self)
    {
        if !self.fragment_shader_extra_data.updated
        {
            return;
        }

        // Update buffer
        let mut buffer_write = self
            .fragment_shader_extra_data
            .buffer
            .write()
            .expect("Failed to lock fragment shaders' extra data buffer for writing");

        *buffer_write = self.fragment_shader_extra_data.data.clone();

        // Fragment shaders' extra shared data is always the same on all shaders,
        // so just use the fallback pipeline to retrieve the layout
        let graphics_pipeline = &self
            .material_pipeline_data
            .get(FALLBACK_MATERIAL_PIPELINE_NAME)
            .expect("Failed retrieving the fallback graphics pipeline")
            .graphics_pipeline;

        let descriptor_set_layout = graphics_pipeline
            .layout()
            .set_layouts()
            .get(FRAGMENT_SHADER_EXTRA_DATA_SET_LAYOUT_INDEX)
            .unwrap()
            .clone();

        self.fragment_shader_extra_data.descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            descriptor_set_layout,
            [WriteDescriptorSet::buffer(0, self.fragment_shader_extra_data.buffer.clone())],
            [],
        )
        .expect("Failed to recreate fragment shaders' extra data descriptor set");

        self.fragment_shader_extra_data.updated = false;
    }

    /// Update environment map lighting data buffer if needed
    /// # Panics
    /// Will panic if this is called while the GPU is using the uniform buffer\
    /// Need to wait until the previous frame is finished processing before calling this function
    fn update_lighting_buffers(&mut self)
    {
        // If data or texture has not changed, don't update anything
        if !self.updated_environment_map_data && !self.updated_environment_map_texture
        {
            return;
        }

        // Check if lighting settings have changed and update buffer
        if self.updated_environment_map_data
        {
            let mut buffer_write = self.environment_map_data_buffer.write().unwrap();
            *buffer_write = self.environment_map_data.clone();
        }

        // Environment map descriptor set is always the same on all shaders, so just update it
        self.environment_map_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            Self::get_environment_map_data_layout(self.device.clone()),
            [
                WriteDescriptorSet::buffer(0, self.environment_map_data_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.environment_texture_data.irradiance_map.image_view.clone(),
                    self.environment_texture_data.irradiance_map.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    self.environment_texture_data.prefiltered_map.image_view.clone(),
                    self.environment_texture_data.prefiltered_map.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    3,
                    self.environment_texture_data.brdf_integration_map.image_view.clone(),
                    self.environment_texture_data.brdf_integration_map.sampler.clone(),
                ),
            ],
            [],
        )
        .expect("Failed to update view projection data buffer");

        // Reset update flags
        self.updated_environment_map_data = false;
        self.updated_environment_map_texture = false;
    }

    /// Update PBR parameters buffer data for all materials if there were any changes
    /// # Panics
    /// Will panic if this is called while the GPU is using the uniform buffer\
    /// Need to wait until the previous frame is finished processing before calling this function
    fn update_pbr_parameters_buffers(&mut self)
    {
        for (material_name, pipeline_data) in self.material_pipeline_data.iter_mut()
        {
            if let Some(ref mut pbr_parameters_data) = pipeline_data.pbr_parameters_data
            {
                // Only update buffers that were actually changed
                if pbr_parameters_data.updated_parameters
                {
                    let mut buffer_write =
                        pbr_parameters_data.pbr_parameters_buffer.write().unwrap();
                    *buffer_write = pbr_parameters_data.pbr_parameters.clone();

                    let descriptor_set_layout = pipeline_data
                        .graphics_pipeline
                        .layout()
                        .set_layouts()
                        .get(PBR_PARAMS_DESCRIPTOR_SET_LAYOUT_INDEX)
                        .expect(
                            format!(
                                "Failed to get PBR parameters descriptor set layout \
                                         for {material_name}"
                            )
                            .as_str(),
                        )
                        .clone();

                    pbr_parameters_data.pbr_parameters_descriptor_set = DescriptorSet::new(
                        self.descriptor_set_allocator.clone(),
                        descriptor_set_layout,
                        [WriteDescriptorSet::buffer(
                            0,
                            pbr_parameters_data.pbr_parameters_buffer.clone(),
                        )],
                        [],
                    )
                    .expect(
                        format!(
                            "Failed to create new PBR parameters \
                                     descriptor set for {material_name}"
                        )
                        .as_str(),
                    );

                    pbr_parameters_data.updated_parameters = false;
                }
            }
        }
    }

    /// Update view-projection buffer data from latest camera update
    /// # Panics
    /// Will panic if this is called while the GPU is using the uniform buffer\
    /// Need to wait until the previous frame is finished processing before calling this function
    fn update_view_projection_buffer(&mut self)
    {
        let mut buffer_write = self.view_projection_data_buffer.write().unwrap();
        *buffer_write = self.view_projection_data.clone();

        // View projection descriptor set is always the same on all shaders,
        // so just use the fallback pipeline to retrieve the layout
        let graphics_pipeline = &self
            .material_pipeline_data
            .get(FALLBACK_MATERIAL_PIPELINE_NAME)
            .expect("Failed retrieving fallback graphics pipeline")
            .graphics_pipeline;

        let descriptor_set_layout = graphics_pipeline
            .layout()
            .set_layouts()
            .get(VIEW_PROJ_DESCRIPTOR_SET_LAYOUT_INDEX)
            .unwrap()
            .clone();

        self.view_projection_data_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            descriptor_set_layout,
            [WriteDescriptorSet::buffer(0, self.view_projection_data_buffer.clone())],
            [],
        )
        .expect("Failed to update view projection data buffer");
    }

    /// Recreate all material and post process graphics pipelines using the current render pass
    fn recreate_graphics_pipelines(&mut self)
    {
        // Recreate material graphics pipelines
        for (_, material_pipeline_data) in self.material_pipeline_data.iter_mut()
        {
            let vert_entry_point = Self::get_shader_entry_point(
                &self.device,
                &material_pipeline_data.vert_shader_path,
                &mut self.shader_entry_points,
            );

            let frag_entry_point = Self::get_shader_entry_point(
                &self.device,
                &material_pipeline_data.frag_shader_path,
                &mut self.shader_entry_points,
            );

            let new_pipeline = Self::create_material_graphics_pipeline(
                &self.device,
                &self.swapchain,
                &self.render_pass,
                vert_entry_point,
                frag_entry_point,
                material_pipeline_data.texture_assignments.len() as u32,
                material_pipeline_data.lighting_mode,
            );

            material_pipeline_data.graphics_pipeline = new_pipeline;
        }

        // Recreate post process graphics pipelines
        for (i, post_process_pipeline_data) in self.post_processing_pipelines.iter_mut().enumerate()
        {
            let vert_entry_point = Self::get_shader_entry_point(
                &self.device,
                POST_PROCESS_VERTEX_SHADER_PATH,
                &mut self.shader_entry_points,
            );

            let frag_entry_point = Self::get_shader_entry_point(
                &self.device,
                &post_process_pipeline_data.shader_path,
                &mut self.shader_entry_points,
            );

            let new_pipeline = Self::create_post_process_graphics_pipeline(
                &self.device,
                &self.swapchain,
                &self.render_pass,
                vert_entry_point,
                frag_entry_point,
                (i + 1) as u32, // Post process subpasses start from index 1
            );

            post_process_pipeline_data.graphics_pipeline = Some(new_pipeline);
        }
    }

    /// Recreate render pass using the currently registered post processes' pipeline data
    /// All graphics pipelines will be recreated after the render pass
    fn recreate_render_pass(&mut self)
    {
        // Register default LDR color and depth-stencil attachments
        let color_ldr_attachment = AttachmentDescription {
            format: self.swapchain.image_format(),
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            // Transition to "PresentSrc" seems to be handled automatically by vulkano
            final_layout: ImageLayout::ColorAttachmentOptimal,
            ..AttachmentDescription::default()
        };

        let depth_stencil_attachment = AttachmentDescription {
            format: self.depth_stencil_format,
            load_op: AttachmentLoadOp::Clear,
            final_layout: ImageLayout::DepthStencilAttachmentOptimal,
            ..AttachmentDescription::default()
        };

        let mut attachments = Vec::from([color_ldr_attachment.clone(), depth_stencil_attachment]);

        let color_ldr_attachment_ref = AttachmentReference {
            attachment: 0,
            layout: ImageLayout::ColorAttachmentOptimal,
            ..AttachmentReference::default()
        };

        let depth_stencil_attachment_ref = AttachmentReference {
            attachment: 1,
            layout: ImageLayout::DepthStencilAttachmentOptimal,
            ..AttachmentReference::default()
        };

        // Register subpasses and dependencies (if any)
        let (subpasses, dependencies) = {
            let mut subpass_descriptions = Vec::new();
            let mut dependencies = Vec::new();

            // If no post processes are registered, then LDR color + depth will be used
            // as the only subpass. Otherwise, LDR color will be the final subpass without depth
            let post_process_num = self.post_processing_pipelines.len();
            if post_process_num == 0
            {
                let single_subpass = SubpassDescription {
                    color_attachments: Vec::from([Some(color_ldr_attachment_ref.clone())]),
                    depth_stencil_attachment: Some(depth_stencil_attachment_ref),
                    ..SubpassDescription::default()
                };

                subpass_descriptions.push(single_subpass)
            }
            else
            {
                // Initial HDR color + depth stencil subpass
                let color_hdr_attachment = AttachmentDescription {
                    format: HDR_FORMAT,
                    load_op: AttachmentLoadOp::Clear,
                    final_layout: ImageLayout::ColorAttachmentOptimal,
                    ..AttachmentDescription::default()
                };

                attachments.push(color_hdr_attachment.clone());

                let color_hdr_attachment_ref = AttachmentReference {
                    attachment: (attachments.len() - 1) as u32,
                    layout: ImageLayout::ColorAttachmentOptimal,
                    ..AttachmentReference::default()
                };

                let first_subpass = SubpassDescription {
                    color_attachments: Vec::from([Some(color_hdr_attachment_ref.clone())]),
                    depth_stencil_attachment: Some(depth_stencil_attachment_ref),
                    ..SubpassDescription::default()
                };

                subpass_descriptions.push(first_subpass);

                // If there are more than one post process, then need to create an extra
                // output HDR color attachment
                if post_process_num > 1
                {
                    attachments.push(color_hdr_attachment.clone());
                }

                // Add each post process subpass
                let mut next_output_attachment_index = 3;

                for i in 0..post_process_num
                {
                    // Last post process outputs to the LDR color attachment
                    // All others output to the intermediate HDR color attachments (alternated
                    // between subpasses)
                    let post_process_color_attachment_index = {
                        if i == post_process_num - 1 { 0 } else { next_output_attachment_index }
                    };

                    let post_process_color_attachment_ref = AttachmentReference {
                        attachment: post_process_color_attachment_index,
                        layout: ImageLayout::ColorAttachmentOptimal,
                        ..AttachmentReference::default()
                    };

                    let post_process_input_attachment_ref = AttachmentReference {
                        attachment: if next_output_attachment_index == 3 { 2 } else { 3 },
                        layout: ImageLayout::ShaderReadOnlyOptimal,
                        ..AttachmentReference::default()
                    };

                    let post_process_subpass = SubpassDescription {
                        color_attachments: Vec::from([Some(post_process_color_attachment_ref)]),
                        input_attachments: Vec::from([Some(post_process_input_attachment_ref)]),
                        ..SubpassDescription::default()
                    };

                    let subpass_dependency = SubpassDependency {
                        src_subpass: Some(i as u32),
                        dst_subpass: Some((i + 1) as u32),
                        src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                        dst_stages: PipelineStages::FRAGMENT_SHADER,
                        src_access: AccessFlags::COLOR_ATTACHMENT_READ,
                        dst_access: AccessFlags::SHADER_WRITE,
                        ..SubpassDependency::default()
                    };

                    subpass_descriptions.push(post_process_subpass);
                    dependencies.push(subpass_dependency);

                    // Alternate HDR color attachments between input and output
                    next_output_attachment_index =
                        if next_output_attachment_index == 3 { 2 } else { 3 };
                }
            }

            (subpass_descriptions, dependencies)
        };

        let render_pass_create_info = RenderPassCreateInfo {
            attachments,
            subpasses,
            dependencies,
            ..RenderPassCreateInfo::default()
        };

        self.render_pass = RenderPass::new(self.device.clone(), render_pass_create_info)
            .expect("Failed to recreate render pass");

        self.recreate_graphics_pipelines();
    }

    /// Recreate all registered post processes' attachments and descriptor sets
    /// using the current swapchain image extents
    fn recreate_post_process_attachments(&mut self)
    {
        for pipeline_data in self.post_processing_pipelines.iter_mut()
        {
            if let Some(ref graphics_pipeline) = pipeline_data.graphics_pipeline
            {
                // Create one set of attachments per swapchain image
                pipeline_data.attachments_image_views = Vec::new();
                pipeline_data.attachments_descriptor_sets = Vec::new();

                let image_extent = self.swapchain.image_extent();
                for _ in 0..self.swapchain.image_count()
                {
                    let (image_views, descriptor_set) = Self::create_post_process_attachments(
                        self.memory_allocator.clone(),
                        self.descriptor_set_allocator.clone(),
                        graphics_pipeline.layout().set_layouts()[0].clone(),
                        [image_extent[0], image_extent[1], 1],
                        true,
                    );

                    pipeline_data.attachments_image_views.push(image_views);
                    pipeline_data.attachments_descriptor_sets.push(descriptor_set);
                }
            }
            else
            {
                Self::log_error(
                    format!(
                        "Graphics pipeline for post process with shader {} is \"None\".\
                        Unable to create post process attachments.",
                        pipeline_data.shader_path,
                    )
                    .as_str(),
                );
            }
        }
    }

    /// Recreate swapchain to match new surface's properties
    fn recreate_swapchain(&mut self)
    {
        // Recreate swapchain to match new window size
        let swapchain_create_info =
            SwapchainCreateInfo { image_extent: self.window_size, ..self.swapchain.create_info() };

        let (new_swapchain, new_swapchain_images) =
            self.swapchain.recreate(swapchain_create_info).expect("Failed to recreate swapchain");

        self.swapchain = new_swapchain;

        // Recreate all post process attachments (to match new swapchain size)
        // then gather all subpasses attachments for framebuffers
        self.recreate_post_process_attachments();

        let mut subpasses_attachments = Vec::new();
        for post_process in &self.post_processing_pipelines
        {
            subpasses_attachments.push(post_process.attachments_image_views.clone());
        }

        // Need to recreate framebuffers as well so that the swapchain references are updated
        self.framebuffers = Self::create_framebuffers(
            self.memory_allocator.clone(),
            &new_swapchain_images,
            &self.render_pass,
            &subpasses_attachments,
            self.depth_stencil_format,
        );

        // Update viewport size
        self.viewport.extent = self.window_size.map(|value| value as f32);

        // Mark as no longer needing update
        self.should_recreate_swapchain = false;
    }
}
