//! GPU side state.

use cuda_std::prelude::*;

/// GPU Side state of all threads.
///
/// This struct stores the following properties for each thread:
/// * Current `i`, `j` positions in the adjacency matrix.
/// * Edge buffer
///   * Array of left nodes.
///   * Array of right nodes.
///   * Counter of how many edges in thread's buffer.
/// * Whether the thread is `done`.
/// Additionally, the following shared properties are stored:
/// * Number of threads.
/// * Capacity of the edge buffers. (All edge buffers are equally big.)
///
/// These properties are stored in the form of one array for each property, indexed by the thread id.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GPUThreadState {
    /// Array of `i` positions.
    pub current_x: *mut u64,
    /// Array of `j` positions.
    pub current_y: *mut u64,
    /// Edge buffer left nodes.
    pub edges_s: *mut u64,
    /// Edge buffer right nodes.
    pub edges_t: *mut u64,
    /// Capacity of edge buffers.
    pub edges_size: u64,
    /// Number of edges stored in the edge buffers.
    pub edges_count: *mut u64,
    /// Done state of threads.
    pub done: *mut bool,
    /// Total number of threads.
    pub num_threads: u64,
    /// Debug value that can be passed back and forth to the CPU.
    pub debug: *mut f32,
}

#[cfg(not(target_os = "cuda"))]
unsafe impl cust::memory::DeviceCopy for GPUThreadState {}

impl GPUThreadState {
    /// Set the current `i` value for the current thread.
    pub unsafe fn set_x(&mut self, x: u64) {
        *self.current_x.add(thread::index_1d() as usize) = x;
    }

    /// Get the current `i` value for the current thread.
    pub unsafe fn get_x(&self) -> u64 {
        *self.current_x.add(thread::index_1d() as usize)
    }

    /// Set the current `j` value for the current thread.
    pub unsafe fn set_y(&mut self, y: u64) {
        *self.current_y.add(thread::index_1d() as usize) = y;
    }

    /// Get the current `j` value for the current thread.
    pub unsafe fn get_y(&self) -> u64 {
        *self.current_y.add(thread::index_1d() as usize)
    }

    /// Set the `done` value for the current thread.
    pub unsafe fn set_done(&mut self, y: bool) {
        *self.done.add(thread::index_1d() as usize) = y;
    }

    /// Get the `done` value for the current thread.
    pub unsafe fn get_done(&self) -> bool {
        *self.done.add(thread::index_1d() as usize)
    }

    /// Get number of edges in this thread's edge buffer.
    pub unsafe fn get_edge_count(&self) -> u64 {
        *self.edges_count.add(thread::index_1d() as usize)
    }

    /// Increment the number of edges stored for this thread by the given amount.
    unsafe fn increment_edge_count(&mut self, count: u64) {
        *self.edges_count.add(thread::index_1d() as usize) += count;
    }

    /// Check if this thread's edge buffer has space left.
    pub unsafe fn can_add_edge(&self) -> bool {
        self.get_edge_count() < self.edges_size
    }

    /// Add edge to this thread's edge buffer. (YOU HAVE TO CHECK IF THERE IS SPACE FIRST)
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
