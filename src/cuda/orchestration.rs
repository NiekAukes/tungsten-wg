use crate::{cuda::{CudaModule, FunctionRef, Type}, spmt::model::Interned};

pub struct CudaOrchestration<'m> {
    pub module: CudaModule<'m>,
    /// Pre-arranged waves (reuse Orchestration::arrange_waves_for()).
    pub waves: Vec<Vec<KernelNodeRef<'m>>>,
    pub arena: &'m bumpalo::Bump,
}

pub struct KernelNode<'m> {
    pub kernel: FunctionRef<'m>,
    pub launch_config: LaunchConfig,
    pub input_buffers: Vec<BufferBinding<'m>>,
    pub output_buffer: BufferDescriptor,
    // NVIDIA debugging hooks:
    pub nvtx_name: Option<String>,   // Nsight Systems timeline label
    pub stream_index: usize,         // which cudaStream_t
    pub emit_events: bool,           // whether to record cudaEvent_t start/end
}

pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_memory_bytes: usize,
}

pub struct BufferDescriptor {
    pub name: String,
    pub element_type: Type,
    pub num_elements: usize,
}

pub struct BufferBinding<'m> {
    pub source_node: KernelNodeRef<'m>,
    pub param_name: String,   // which kernel parameter receives this pointer
    pub element_offset: usize,
    pub num_elements: usize,
}

pub type KernelNodeRef<'m> = Interned<'m, KernelNode<'m>>;