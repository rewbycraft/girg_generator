use cuda_std::prelude::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GPUThreadState {
    pub current_x: *mut u64,
    pub current_y: *mut u64,
    pub edges_s: *mut u64,
    pub edges_t: *mut u64,
    pub edges_size: u64,
    pub edges_count: *mut u64,
    pub done: *mut bool,
    pub num_threads: u64,
    pub debug: *mut f32,
}

#[cfg(not(target_os = "cuda"))]
unsafe impl cust::memory::DeviceCopy for GPUThreadState {}

impl GPUThreadState {
    pub unsafe fn set_x(&mut self, x: u64) {
        *self.current_x.add(thread::index_1d() as usize) = x;
    }

    pub unsafe fn get_x(&self) -> u64 {
        *self.current_x.add(thread::index_1d() as usize)
    }

    pub unsafe fn set_y(&mut self, y: u64) {
        *self.current_y.add(thread::index_1d() as usize) = y;
    }

    pub unsafe fn get_y(&self) -> u64 {
        *self.current_y.add(thread::index_1d() as usize)
    }

    pub unsafe fn set_done(&mut self, y: bool) {
        *self.done.add(thread::index_1d() as usize) = y;
    }

    pub unsafe fn get_done(&self) -> bool {
        *self.done.add(thread::index_1d() as usize)
    }

    pub unsafe fn get_edge_count(&self) -> u64 {
        *self.edges_count.add(thread::index_1d() as usize)
    }

    unsafe fn increment_edge_count(&mut self, count: u64) {
        *self.edges_count.add(thread::index_1d() as usize) += count;
    }

    pub unsafe fn can_add_edge(&self) -> bool {
        self.get_edge_count() < self.edges_size
    }

    pub unsafe fn add_edge(&mut self, s: u64, t: u64) {
        //TODO: Add safety check
        if !self.can_add_edge() {
            panic!("cannot add edge")
        }

        let id = thread::index_1d() as usize;
        let pos = (id * (self.edges_size as usize)) + (self.get_edge_count() as usize);

        *self.edges_s.add(pos) = s;
        *self.edges_t.add(pos) = t;

        self.increment_edge_count(1);
    }

}
