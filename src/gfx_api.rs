//! Graphics API that will be used to render onto the application's window

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use glam::Mat4;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, RenderPassBeginInfo};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::image::{Image, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::library::VulkanLibrary;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::{DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::{EntryPoint, ShaderModule, ShaderModuleCreateInfo, ShaderStages};
use vulkano::swapchain::{self, CompositeAlpha, PresentFuture, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, Version, VulkanError};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture, NowFuture};
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

/// Object's transform data (passed as a push constant)
#[derive(BufferContents, Clone, Default)]
#[repr(C)]
struct ObjectTransformData
{
    matrix: [[f32; 4]; 4],
}

/// Model vertex buffer data
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct VertexData
{
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],

    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

/// View + perspective projection matrices (passed as a uniform buffer)
#[derive(BufferContents, Clone, Default)]
#[repr(C)]
struct ViewProjectionData
{
    view_matrix: [[f32; 4]; 4],
    perspective_matrix: [[f32; 4]; 4],
}

/// Graphics API interface
/// Only Vulkan is supported at the moment
pub struct GfxApi
{
    // Common application objects that are accessed at many different points of the application
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    // Rendering data that will be used in the shaders
    vertex_buffer: Subbuffer<[VertexData]>, // Hard-coded triangle vertex buffer
    object_transform_data: ObjectTransformData, // Hard-coded object transform buffer
    view_projection_data: ViewProjectionData, // Current camera (view-projection) data
    view_perspective_data_buffer: Subbuffer<ViewProjectionData>,
    view_perspective_data_descriptor_set: Arc<DescriptorSet>,

    // Data related to presenting onto the surface each frame
    should_recreate_swapchain: bool,
    window_size: [u32; 2],
    previous_frame_end: Option<FenceSignalFuture<PresentFuture<CommandBufferExecFuture<JoinFuture<NowFuture, SwapchainAcquireFuture>>>>>,
    // ^ Need this ridiculous nested type to be able to use FenceSignalFuture's API
    // TODO: Might be able to improve it through different API?
}

impl GfxApi
{
    /// Create and initialize a new graphics API instance
    /// API will be ready after being created, only need to call [render] method
    pub fn new(event_loop: &ActiveEventLoop, window: Arc<Window>) -> Self
    {
        // Init Vulkan library
        let library = VulkanLibrary::new().expect("Vulkan library creation failed");

        // Find required extensions for winit window and initialize instance
        let required_extensions = Surface::required_extensions(event_loop)
            .expect("Failed to retrieve required extensions for Vulkan from winit window");

        #[cfg(debug_assertions)]
        Self::log_required_extensions(&required_extensions);

        let instance_create_info = InstanceCreateInfo
        {
            application_name: Some(String::from("Rusty Engine")),
            application_version: Version::major_minor(1, 0),
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            enabled_layers: Self::get_enabled_layers(),
            ..Default::default()
        };

        let instance = Instance::new(library, instance_create_info)
            .expect("Failed to create Vulkan instance");

        // Find the best physical device candidate and create a logical device for it
        let device_extensions = DeviceExtensions
        {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            GfxApi::get_best_physical_device_and_queue(&instance, &event_loop, &device_extensions);

        let physical_device_properties = physical_device.properties();

        #[cfg(debug_assertions)]
        println!("Initializing logical device using {} :: {}",
                 physical_device_properties.device_name,
                 Self::get_device_type_name(&physical_device_properties.device_type));

        // Initialize logical device from physical device and store queue for later
        let device_create_info = DeviceCreateInfo
        {
            enabled_extensions: device_extensions,
            queue_create_infos: Vec::from(
                [QueueCreateInfo { queue_family_index, ..Default::default() }]
            ),
            ..Default::default()
        };

        let (device, mut queues) = Device::new(physical_device, device_create_info)
            .expect("Failed to create logical device");
        let queue = queues.next().unwrap(); // Using first queue only for now

        // Create surface and swapchain
        let surface = Surface::from_window(instance.clone(), window.clone())
            .expect("Failed  to create surface");

        let min_image_count = device
            .physical_device()
            .surface_capabilities(&surface, Default::default()).unwrap()
            .min_image_count.max(2); // Need at least two for fullscreen

        let (image_format, _) = device
            .physical_device()
            .surface_formats(&surface, Default::default()).unwrap()[0];

        let swapchain_create_info = SwapchainCreateInfo
        {
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

        // Load shaders
        let vert_shader_path = "src/shaders/vert.spv".replace("/", "\\");
        let frag_shader_path = "src/shaders/frag.spv".replace("/", "\\");

        let (vert_shader_entry_point, frag_shader_entry_point) =
            Self::load_shaders(device.clone(), vert_shader_path, frag_shader_path);

        // Create single pass render pass
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: { color: [color], depth_stencil: {} },
        )
        .unwrap();

        // Create viewport
        let viewport = Viewport
        {
            extent: window.inner_size().into(),
            ..Default::default()
        };

        // Create graphics pipeline
        let graphics_pipeline = Self::create_graphics_pipeline(
            &device,
            &swapchain,
            &render_pass,
            vert_shader_entry_point,
            frag_shader_entry_point,
        );

        // Create matching framebuffers for swapchain images
        let framebuffers = Self::create_framebuffers(&swapchain_images, &render_pass);

        // Create memory allocators
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default()
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Allocate hard-coded triangle mesh and set position
        let vertex_buffer = Self::allocate_vertex_buffer(memory_allocator.clone());

        let object_transform_data =
            ObjectTransformData { matrix: Mat4::IDENTITY.to_cols_array_2d() };

        // Allocate view and perspective projection data
        let view_projection_data = ViewProjectionData::default();
        let view_perspective_data_buffer =
            Self::allocate_view_perspective_data_buffer(memory_allocator, view_projection_data.clone());

        let view_perspective_data_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            graphics_pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, view_perspective_data_buffer.clone())],
            []
        )
        .unwrap();

        GfxApi {
            device,
            queue,
            swapchain,
            viewport,
            render_pass,
            graphics_pipeline,
            framebuffers,
            command_buffer_allocator,
            descriptor_set_allocator,
            vertex_buffer,
            object_transform_data,
            view_projection_data,
            view_perspective_data_buffer,
            view_perspective_data_descriptor_set,
            should_recreate_swapchain: false,
            window_size: window.inner_size().into(),
            previous_frame_end: None,
        }
    }

    /// Notify graphics API that window has been resized so that it updates its rendering
    /// state to match the new window size
    pub fn on_window_resized(&mut self, new_size: PhysicalSize<u32>)
    {
        self.should_recreate_swapchain = true;
        self.window_size = new_size.into();
    }

    /// Record and execute rendering commands once the next frame image is ready
    pub fn render(&mut self)
    {
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
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        )
        .expect("Failed to create command buffer builder");

        let render_pass_begin_info = RenderPassBeginInfo
        {
            clear_values: Vec::from([Some([0.1, 0.1, 0.1, 1.0].into())]),
            ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index as usize].clone())
        };

        command_buffer_builder.begin_render_pass(
            render_pass_begin_info,
            Default::default()
        )
        .unwrap()
        .set_viewport(0, [self.viewport.clone()].into_iter().collect()).unwrap()
        .bind_pipeline_graphics(self.graphics_pipeline.clone()).unwrap()
        .bind_vertex_buffers(0, self.vertex_buffer.clone()).unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.graphics_pipeline.layout().clone(),
            0,
            self.view_perspective_data_descriptor_set.clone()
        )
        .unwrap()
        .push_constants(self.graphics_pipeline.layout().clone(), 0, self.object_transform_data.clone()).ok();

        unsafe { command_buffer_builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }.unwrap()
            .end_render_pass(Default::default()).ok();

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

        let swapchain_present_info =
            SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index);

        // Execute command buffer, present, and then signal fence
        let present_fence = sync::now(self.device.clone())
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), swapchain_present_info)
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
                    println!("Failed to flush present fence: {err}");
                    self.previous_frame_end = None;
                }
        }
    }

    /// Update current view-perspective matrices for rendering
    pub fn update_camera(&mut self, view_matrix: [[f32; 4]; 4], perspective_matrix: [[f32; 4]; 4])
    {
        // Need to flip Y axis of projection matrix to match Vulkan's viewport coordinates
        // (glam library calculates a perspective matrix meant for OpenGL)
        let mut flipped_perspective_matrix = perspective_matrix;
        flipped_perspective_matrix[1][1] *= -1.0;

        self.view_projection_data = ViewProjectionData
        {
            view_matrix,
            perspective_matrix: flipped_perspective_matrix,
        };
    }

    /// Update view-projection buffer data from latest camera update
    /// # Panics
    /// Will panic if this is called while the GPU is using the uniform buffer
    /// Need to wait until the previous frame is finished processing before calling this function
    fn update_view_projection_buffer(&mut self)
    {
        let mut buffer_write = self.view_perspective_data_buffer.write().unwrap();
        *buffer_write = self.view_projection_data.clone();

        self.view_perspective_data_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.graphics_pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, self.view_perspective_data_buffer.clone())],
            []
        )
        .expect("Failed to update view projection data buffer");
    }

    /// Find the best physical device that supports required extensions
    fn get_best_physical_device_and_queue(
        instance: &Arc<Instance>,
        event_loop: &ActiveEventLoop,
        device_extensions: &DeviceExtensions)
        -> (Arc<PhysicalDevice>, u32)
    {
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter_map(|device| {
                // Check if it supports required extensions and queue family
                if device.supported_extensions().contains(&device_extensions)
                {
                    device.queue_family_properties().iter().enumerate()
                        .position(|(i, queue)| {
                            queue.queue_flags.intersects(QueueFlags::GRAPHICS) &&
                                device.presentation_support(i as u32, event_loop).unwrap()
                        })
                        .map(|index| (device, index as u32))
                }
                else
                {
                    None
                }
            })
            .min_by_key(|(device, _)| {
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

        (physical_device, queue_family_index)
    }

    /// Get validation layers to be used depending on build configuration
    fn get_enabled_layers() -> Vec<String>
    {
        if cfg!(debug_assertions)
        {
            return Vec::from([String::from("VK_LAYER_KHRONOS_validation")])
        }

        Vec::new()
    }

    /// Allocate hard-coded triangle vertex buffer using given [MemoryAllocator]
    fn allocate_vertex_buffer(memory_allocator: Arc<dyn MemoryAllocator>) -> Subbuffer<[VertexData]>
    {
        // Hard-code triangle vertices for now
        let vertices =
        [
            VertexData { position: [0.5, -0.5, 0.0], color: [1.0, 0.0, 0.0] },
            VertexData { position: [0.5, 0.5, 0.0], color: [0.0, 1.0, 0.0] },
            VertexData { position: [-0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
        ];

        let buffer_create_info = BufferCreateInfo
        {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        };

        let allocation_create_info = AllocationCreateInfo
        {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        Buffer::from_iter(
            memory_allocator,
            buffer_create_info,
            allocation_create_info,
            vertices
        )
        .unwrap()
    }

    /// Allocate view + perspective projection matrices data using given [MemoryAllocator]
    fn allocate_view_perspective_data_buffer(
        memory_allocator: Arc<dyn MemoryAllocator>,
        view_projection_data: ViewProjectionData)
        -> Subbuffer<ViewProjectionData>
    {
        let buffer_create_info = BufferCreateInfo
        {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        };

        let allocation_create_info = AllocationCreateInfo
        {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        Buffer::from_data(
            memory_allocator,
            buffer_create_info,
            allocation_create_info,
            view_projection_data,
        )
        .unwrap()
    }

    /// Create and initialize the graphics pipeline that will be used for rendering
    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swapchain: &Arc<Swapchain>,
        render_pass: &Arc<RenderPass>,
        vert_shader_entry_point: EntryPoint,
        frag_shader_entry_point: EntryPoint)
        -> Arc<GraphicsPipeline>
    {
        let vertex_input_state = VertexData::per_vertex()
            .definition(&vert_shader_entry_point)
            .unwrap();

        let scissor = Scissor
        {
            offset: [0; 2],
            extent: swapchain.image_extent(),
        };

        let viewport_state = ViewportState
        {
            scissors: [scissor].into_iter().collect(),
            ..Default::default() // Not setting viewport since it is resized dynamically
        };

        let rasterization_state = RasterizationState
        {
            cull_mode: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            ..Default::default()
        };

        let color_blend_state = ColorBlendState
        {
            attachments: Vec::from([ColorBlendAttachmentState::default()]),
            ..Default::default()
        };

        let subpass = Subpass::from(render_pass.clone(), 0).expect("Failed to create subpass");

        let pipeline_stages = [
            PipelineShaderStageCreateInfo::new(vert_shader_entry_point),
            PipelineShaderStageCreateInfo::new(frag_shader_entry_point),
        ];

        // Declare push constant range for objects' transform matrix,
        // and descriptor set layout for view and perspective matrices,
        // then pass to pipeline layout
        let push_constant_range = PushConstantRange
        {
            stages: ShaderStages::VERTEX,
            offset: 0,
            size: size_of::<ObjectTransformData>() as u32,
        };

        let descriptor_set_layout_binding = DescriptorSetLayoutBinding
        {
            descriptor_type: DescriptorType::UniformBuffer,
            descriptor_count: 1,
            stages: ShaderStages::VERTEX,
            ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
        };

        let descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo
        {
            bindings: BTreeMap::from([(0, descriptor_set_layout_binding)]),
            ..DescriptorSetLayoutCreateInfo::default()
        };

        let descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), descriptor_set_layout_create_info).unwrap();

        let pipeline_layout_create_info = PipelineLayoutCreateInfo
        {
            push_constant_ranges: Vec::from([push_constant_range]),
            set_layouts: Vec::from([descriptor_set_layout]),
            ..PipelineLayoutCreateInfo::default()
        };

        let pipeline_layout =
            PipelineLayout::new(device.clone(), pipeline_layout_create_info).unwrap();

        let graphics_pipeline_create_info = GraphicsPipelineCreateInfo
        {
            stages: pipeline_stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
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

        GraphicsPipeline::new(
            device.clone(),
            None,
            graphics_pipeline_create_info,
        )
        .expect("Failed to create graphics pipeline")
    }

    /// Create a framebuffer per image and making them compatible with the given render pass
    fn create_framebuffers(
        images: &Vec<Arc<Image>>,
        render_pass: &Arc<RenderPass>)
        -> Vec<Arc<Framebuffer>>
    {
        let mut framebuffers = Vec::new();

        for image in images
        {
            let image_view = ImageView::new_default(image.clone()).unwrap();

            let framebuffer_create_info = FramebufferCreateInfo
            {
                attachments: Vec::from([image_view]),
                ..Default::default()
            };

            let framebuffer =
                Framebuffer::new(render_pass.clone(), framebuffer_create_info).expect("Failed to create framebuffer");

            framebuffers.push(framebuffer);
        }

        framebuffers
    }

    /// Load vertex and fragment shader at the given paths
    ///
    /// # Panics
    /// It will panic if any of the files doesn't exist, or if they are not correct shader files
    fn load_shaders(
        device: Arc<Device>,
        vert_shader_path: impl AsRef<Path>,
        frag_shader_path: impl AsRef<Path>)
        -> (EntryPoint, EntryPoint)
    {
        let paths = Vec::from([vert_shader_path.as_ref(), frag_shader_path.as_ref()]);
        let mut entry_points = Vec::new();

        for path in paths
        {
            let filepath = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);


            let mut bytes = Vec::new();
            let mut file = File::open(&filepath)
                .expect(&format!("Failed to open shader file {}", filepath.to_str().unwrap()));

            file.read_to_end(&mut bytes)
                .expect(&format!("Failed to read shader file {}", filepath.to_str().unwrap()));

            let shader_words = vulkano::shader::spirv::bytes_to_words(&bytes)
                .expect("Failed to read shader words from file {filepath}");

            let shader_module_create_info = ShaderModuleCreateInfo::new(&shader_words);
            let shader_module = unsafe
                {
                    ShaderModule::new(device.clone(), shader_module_create_info).unwrap()
                };

            let entry_point = shader_module.entry_point("main").unwrap();
            entry_points.push(entry_point);
        }

        let frag_entry_point = entry_points.pop().unwrap();
        let vert_entry_point = entry_points.pop().unwrap();

        (vert_entry_point, frag_entry_point)
    }

    /// Recreate swapchain to match new surface's properties
    fn recreate_swapchain(&mut self)
    {
        // Recreate swapchain to match new window size
        let swapchain_create_info = SwapchainCreateInfo
        {
            image_extent: self.window_size,
            ..self.swapchain.create_info()
        };

        let (new_swapchain, new_swapchain_images) =
            self.swapchain.recreate(swapchain_create_info).expect("Failed to recreate swapchain");

        self.swapchain = new_swapchain;

        // Need to recreate framebuffers as well so that the swapchain references are updated
        self.framebuffers = Self::create_framebuffers(&new_swapchain_images, &self.render_pass);

        // Update viewport size
        self.viewport.extent = self.window_size.map(|value| value as f32);

        // Mark as no longer needing update
        self.should_recreate_swapchain = false;
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
}