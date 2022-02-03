use cust::error::CudaResult;
use cust::memory::*;
use cust::prelude::*;
use derivative::Derivative;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct CPUThreadState {
    pub current_x: Vec<u64>,
    #[derivative(Debug = "ignore")]
    current_x_d: DeviceBuffer<u64>,
    pub current_y: Vec<u64>,
    #[derivative(Debug = "ignore")]
    current_y_d: DeviceBuffer<u64>,
    pub edges_s: Vec<u64>,
    #[derivative(Debug = "ignore")]
    edges_s_d: DeviceBuffer<u64>,
    pub edges_t: Vec<u64>,
    #[derivative(Debug = "ignore")]
    edges_t_d: DeviceBuffer<u64>,
    pub edges_size: u64,
    pub edges_count: Vec<u64>,
    #[derivative(Debug = "ignore")]
    edges_count_d: DeviceBuffer<u64>,
    pub done: Vec<bool>,
    #[derivative(Debug = "ignore")]
    done_d: DeviceBuffer<bool>,
    pub num_threads: u64,

    pub debug: Vec<f32>,
    #[derivative(Debug = "ignore")]
    debug_d: DeviceBuffer<f32>,
}

impl CPUThreadState {
    pub fn new(edges_size: u64, num_threads: u64) -> CudaResult<Self> {
        let mut s = Self {
            current_x: Vec::new(),
            current_x_d: unsafe { DeviceBuffer::zeroed(num_threads as usize)? },
            current_y: Vec::new(),
            current_y_d: unsafe { DeviceBuffer::zeroed(num_threads as usize)? },
            edges_s: Vec::new(),
            edges_s_d: unsafe {
                DeviceBuffer::zeroed((num_threads as usize) * (edges_size as usize))?
            },
            edges_t: Vec::new(),
            edges_t_d: unsafe {
                DeviceBuffer::zeroed((num_threads as usize) * (edges_size as usize))?
            },
            edges_size,
            edges_count: Vec::new(),
            edges_count_d: unsafe { DeviceBuffer::zeroed(num_threads as usize)? },
            done: Vec::new(),
            done_d: unsafe { DeviceBuffer::zeroed(num_threads as usize)? },
            debug: Vec::new(),
            debug_d: unsafe { DeviceBuffer::zeroed(10)? },
            num_threads,
        };
        s.current_x.resize(num_threads as usize, 0);
        s.current_y.resize(num_threads as usize, 0);
        s.edges_count.resize(num_threads as usize, 0);
        s.done.resize(num_threads as usize, false);
        s.edges_s
            .resize((num_threads as usize) * (edges_size as usize), 0);
        s.edges_t
            .resize((num_threads as usize) * (edges_size as usize), 0);
        s.debug.resize(10, 0.0);
        Ok(s)
    }

    pub fn edges_iter(&self, tid: usize) -> impl Iterator<Item = (u64, u64)> + '_ {
        let offset: usize = tid * (self.edges_size as usize);
        let length: usize = self.edges_count[tid].min(self.edges_size) as usize;
        let end = offset + length;

        //let l = self.edges_s.iter().skip(tid * self.edges_size).take(self.edges_count[tid].min(self.edges_size) as usize);
        //let r = self.edges_t.iter().skip(tid * self.edges_size).take(self.edges_count[tid].min(self.edges_size) as usize);
        let l = self.edges_s[offset..end].iter().copied();
        let r = self.edges_t[offset..end].iter().copied();
        l.zip(r)
    }

    pub fn copy_to_device(&mut self) -> CudaResult<()> {
        self.current_x_d.copy_from(&self.current_x)?;
        self.current_y_d.copy_from(&self.current_y)?;
        self.edges_count_d.copy_from(&self.edges_count)?;
        self.done_d.copy_from(&self.done)?;
        self.debug_d.copy_from(&self.debug)?;
        Ok(())
    }

    pub fn copy_from_device(&mut self) -> CudaResult<()> {
        self.current_x_d.copy_to(self.current_x.as_mut_slice())?;
        self.current_y_d.copy_to(self.current_y.as_mut_slice())?;
        self.edges_s_d.copy_to(self.edges_s.as_mut_slice())?;
        self.edges_t_d.copy_to(self.edges_t.as_mut_slice())?;
        self.edges_count_d
            .copy_to(self.edges_count.as_mut_slice())?;
        self.done_d.copy_to(self.done.as_mut_slice())?;
        self.debug_d.copy_to(self.debug.as_mut_slice())?;
        Ok(())
    }

    pub unsafe fn copy_to_device_async(&mut self, stream: &Stream) -> CudaResult<()> {
        self.current_x_d.async_copy_from(&self.current_x, stream)?;
        self.current_y_d.async_copy_from(&self.current_y, stream)?;
        self.edges_count_d
            .async_copy_from(&self.edges_count, stream)?;
        self.done_d.async_copy_from(&self.done, stream)?;
        self.debug_d.async_copy_from(&self.debug, stream)?;
        Ok(())
    }

    pub unsafe fn copy_from_device_async(&mut self, stream: &Stream) -> CudaResult<()> {
        self.current_x_d
            .async_copy_to(self.current_x.as_mut_slice(), stream)?;
        self.current_y_d
            .async_copy_to(self.current_y.as_mut_slice(), stream)?;
        self.edges_s_d
            .async_copy_to(self.edges_s.as_mut_slice(), stream)?;
        self.edges_t_d
            .async_copy_to(self.edges_t.as_mut_slice(), stream)?;
        self.edges_count_d
            .async_copy_to(self.edges_count.as_mut_slice(), stream)?;
        self.done_d
            .async_copy_to(self.done.as_mut_slice(), stream)?;
        self.debug_d
            .async_copy_to(self.debug.as_mut_slice(), stream)?;
        Ok(())
    }

    pub fn create_gpu_state(&self) -> CudaResult<DeviceBox<crate::state::gpu::GPUThreadState>> {
        let s = crate::state::gpu::GPUThreadState {
            current_x: self.current_x_d.as_device_ptr().as_raw_mut(),
            current_y: self.current_y_d.as_device_ptr().as_raw_mut(),
            edges_s: self.edges_s_d.as_device_ptr().as_raw_mut(),
            edges_t: self.edges_t_d.as_device_ptr().as_raw_mut(),
            edges_size: self.edges_size,
            edges_count: self.edges_count_d.as_device_ptr().as_raw_mut(),
            done: self.done_d.as_device_ptr().as_raw_mut(),
            num_threads: self.num_threads,
            debug: self.debug_d.as_device_ptr().as_raw_mut(),
        };
        DeviceBox::new(&s)
    }
}
