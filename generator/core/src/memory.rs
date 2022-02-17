use no_std_compat::slice::{from_raw_parts, from_raw_parts_mut};

#[derive(Copy, Clone, Debug)]
pub struct FixedSizeBuffer<T: Copy> {
    data: *mut T,
    size: u64,
}

#[cfg(all(not(target_os = "cuda"), feature = "gpu"))]
unsafe impl<T: Copy> cust::memory::DeviceCopy for FixedSizeBuffer<T> {}

impl<T: Copy> FixedSizeBuffer<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            from_raw_parts(self.data, self.size as usize)
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            from_raw_parts_mut(self.data, self.size as usize)
        }
    }
}

