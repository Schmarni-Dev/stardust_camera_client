use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use glam::{Mat4, Quat, vec3};
use stardust_xr_cme::{
    dmatex::Dmatex, format::DmatexFormat, render_device::RenderDevice, swapchain::Swapchain,
};
use stardust_xr_fusion::{
    AsyncEventHandle, Client, ClientHandle,
    camera::{Camera, CameraAspect, View},
    drawable::{DmatexSize, DmatexSubmitInfo, MaterialParameter, Model, ModelPartAspect},
    project_local_resources,
    root::{RootAspect, RootEvent},
    spatial::Transform,
    values::ResourceID,
};
use tracing::info;
use vulkano::{
    VulkanLibrary,
    command_buffer::{
        self, AutoCommandBufferBuilder, BlitImageInfo, CommandBufferSubmitInfo, SemaphoreSubmitInfo, SubmitInfo, allocator::StandardCommandBufferAllocator,
    },
    device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags},
    format::Format,
    image::{Image, ImageUsage},
    instance::{Instance, InstanceCreateInfo},
    swapchain::{
        CompositeAlpha, PresentInfo, PresentMode, SemaphorePresentInfo, Surface,
        SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::
        semaphore::Semaphore
    ,
};
use winit::{application::ApplicationHandler, event_loop::EventLoop, window::Window};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();
    info!("Hello, world!");
    let client = Client::connect().await.unwrap();
    client
        .setup_resources(&[&project_local_resources!("res")])
        .unwrap();

    let async_loop = client.async_event_loop();
    let client = async_loop.client_handle.clone();
    let render_dev = RenderDevice::primary_server_device(&client).await.unwrap();
    let library = VulkanLibrary::new().unwrap();

    let event_loop = EventLoop::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: Surface::required_extensions(&event_loop).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    let phys_dev = render_dev.get_physical_device(&instance).unwrap();
    let required_dev_exts = DeviceExtensions {
        khr_swapchain: true,
        ..Default::default()
    } | Dmatex::required_device_exts();
    let required_dev_feats = Dmatex::required_device_features();
    let queue_family_index = phys_dev
        .queue_family_properties()
        .iter()
        .enumerate()
        .find(|(i, p)| {
            p.queue_flags.contains(QueueFlags::TRANSFER)
                && phys_dev
                    .presentation_support(*i as u32, &event_loop)
                    .unwrap()
        })
        .unwrap()
        .0 as u32;
    let (dev, mut queues) = Device::new(
        phys_dev.clone(),
        DeviceCreateInfo {
            enabled_extensions: required_dev_exts,
            enabled_features: required_dev_feats,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();
    let formats = DmatexFormat::enumerate(&client, &render_dev).await.unwrap();
    let cballoc = Arc::new(StandardCommandBufferAllocator::new(
        dev.clone(),
        Default::default(),
    ));
    let output = Arc::<Mutex<Option<Output>>>::default();
    tokio::spawn(stardust_loop(
        async_loop.get_event_handle(),
        client.clone(),
        dev.clone(),
        queue,
        output.clone(),
        cballoc,
    ));
    tokio::task::block_in_place(|| {
        let mut winit_app = WinitApp {
            output,
            dev,
            instance,
            render_dev: render_dev,
            formats,
            client,
        };
        event_loop.run_app(&mut winit_app).unwrap();
    });
}
async fn stardust_loop(
    event: AsyncEventHandle,
    client: Arc<ClientHandle>,
    dev: Arc<Device>,
    queue: Arc<Queue>,
    output: Arc<Mutex<Option<Output>>>,
    cballoc: Arc<StandardCommandBufferAllocator>,
) {
    let camera = Camera::create(
        client.get_root(),
        Transform::from_translation_rotation(
            [0.0, 0.2, 0.2],
            Quat::from_rotation_y(-90f32.to_radians()),
        ),
    )
    .unwrap();
    let model = Model::create(
        &camera,
        Transform::from_scale([0.2; 3]),
        &ResourceID::new_namespaced("vk", "panel"),
    )
    .unwrap();
    let panel = model.part("Panel").unwrap();
    panel
        .set_material_parameter("unlit", MaterialParameter::Bool(true))
        .unwrap();

    loop {
        event.wait().await;
        let _frame_info = match client.get_root().recv_root_event() {
            Some(RootEvent::Ping { response }) => {
                response.send_ok(());
                continue;
            }
            None | Some(RootEvent::SaveState { response: _ }) => {
                continue;
            }
            Some(RootEvent::Frame { info }) => info,
        };
        let output_lock = output.lock().unwrap();
        let Some(output) = output_lock.as_ref() else {
            continue;
        };
        let mut builder = AutoCommandBufferBuilder::primary(
            cballoc.clone(),
            queue.queue_family_index(),
            command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let way_acquire_sema = Arc::new(Semaphore::from_pool(dev.clone()).unwrap());
        let way_release_sema = Arc::new(Semaphore::from_pool(dev.clone()).unwrap());
        let cme_info = output.cme_swapchain.lock().unwrap().prepare_next_image();
        let way_info = unsafe {
            output
                .swapchain
                .acquire_next_image(&vulkano::swapchain::AcquireNextImageInfo {
                    semaphore: Some(way_acquire_sema.clone()),
                    ..Default::default()
                })
                .unwrap()
        };
        let way_image = output.swap_images[way_info.image_index as usize].clone();

        builder
            .blit_image(BlitImageInfo::images(cme_info.image(), way_image))
            .unwrap();
        let cmd_buff = builder.build().unwrap();
        let res = cme_info.image().extent();
        let submit_info = cme_info.submit(&dev, &queue, |wait, mut queue, release| unsafe {
            queue
                .submit(
                    &[SubmitInfo {
                        wait_semaphores: vec![
                            SemaphoreSubmitInfo::new(wait),
                            SemaphoreSubmitInfo::new(way_acquire_sema.clone()),
                        ],
                        command_buffers: vec![CommandBufferSubmitInfo::new(cmd_buff)],
                        signal_semaphores: vec![
                            SemaphoreSubmitInfo::new(release),
                            SemaphoreSubmitInfo::new(way_release_sema.clone()),
                        ],
                        ..Default::default()
                    }],
                    None,
                )
                .unwrap();

            _ = queue
                .present(&PresentInfo {
                    wait_semaphores: vec![SemaphorePresentInfo::new(way_release_sema.clone())],
                    swapchain_infos: vec![SwapchainPresentInfo::swapchain_image_index(
                        output.swapchain.clone(),
                        way_info.image_index,
                    )],
                    ..Default::default()
                })
                .unwrap();
            queue.wait_idle().unwrap();
        });
        let ratio = res[0] as f32 / res[1] as f32;
        // bevy uses reverse Z
        let mat = Mat4::perspective_rh(60f32.to_radians(), ratio, 300.0, 0.003);
        // let mat = Mat4::perspective_infinite_reverse_rh(64f32, ratio, 0.003);

        panel
            .set_material_parameter(
                "diffuse",
                MaterialParameter::Dmatex(DmatexSubmitInfo {
                    dmatex_id: submit_info.dmatex_id,
                    acquire_point: submit_info.release_point,
                    release_point: submit_info.release_point,
                }),
            )
            .unwrap();
        camera
            .request_draw(
                submit_info,
                &[View {
                    projection_matrix: mat.into(),
                    camera_relative_transform: Transform::none(),
                }],
            )
            .unwrap();
    }
}

struct Output {
    _window: Arc<Window>,
    swapchain: Arc<vulkano::swapchain::Swapchain>,
    swap_images: Vec<Arc<Image>>,
    cme_swapchain: Mutex<Swapchain>,
}
struct WinitApp {
    output: Arc<Mutex<Option<Output>>>,
    dev: Arc<Device>,
    instance: Arc<Instance>,
    render_dev: RenderDevice,
    formats: HashMap<Format, DmatexFormat>,
    client: Arc<ClientHandle>,
}
impl ApplicationHandler for WinitApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        info!("creating new window");
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();
        info!(?window_size);

        let (image_format, _) = self
            .dev
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()
            .into_iter()
            .filter(|(f, _)| format!("{:?}", f).contains("SRGB"))
            .next()
            .unwrap();
        let (swapchain, images) = {
            let surface_capabilities = self
                .dev
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            vulkano::swapchain::Swapchain::new(
                self.dev.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format: dbg!(image_format),
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::TRANSFER_DST,
                    composite_alpha: CompositeAlpha::Opaque,
                    present_mode: PresentMode::Mailbox,

                    ..Default::default()
                },
            )
            .unwrap()
        };
        let dmatex_format = self.formats.get(&Format::R8G8B8A8_SRGB).unwrap();
        let cme_swapchain = Swapchain::new(
            &self.client,
            &self.dev,
            &self.render_dev,
            DmatexSize::Dim2D(window_size.into()),
            dmatex_format,
            None,
            ImageUsage::TRANSFER_SRC | ImageUsage::COLOR_ATTACHMENT,
        )
        .into();
        self.output.lock().unwrap().replace(Output {
            _window: window,
            swapchain,
            swap_images: images,
            cme_swapchain,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::Resized(_physical_size) => {}
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            winit::event::WindowEvent::Destroyed => {
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {}
            _ => {}
        }
    }
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // self.output
        //     .lock()
        //     .unwrap()
        //     .as_ref()
        //     .unwrap()
        //     ._window
        //     .request_redraw();
    }
}
