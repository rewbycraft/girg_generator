//TODO: Re-enable missing doc warnings.
//#![warn(missing_docs)]
//#![warn(clippy::missing_docs_in_private_items)]
#![cfg_attr(
target_os = "cuda",
feature(register_attr),
register_attr(nvvm_internal)
)]
#![cfg_attr(not(feature="std"), no_std)]
#![allow(clippy::missing_safety_doc)]

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use no_std_compat::borrow::{Borrow, BorrowMut};
use no_std_compat::slice::{from_raw_parts, from_raw_parts_mut};

#[derive(Copy, Clone, Debug)]
pub struct FixedSizeBufferRef<'a, T: 'a + Copy> {
    data: *const T,
    size: u64,
    _marker: PhantomData<&'a [T]>,
}

#[derive(Copy, Clone, Debug)]
pub struct FixedSizeBufferMut<'a, T: 'a + Copy> {
    data: *mut T,
    size: u64,
    _marker: PhantomData<&'a [T]>,
}

#[cfg(all(not(target_os = "cuda"), feature = "gpu"))]
unsafe impl<'a, T: 'a + Copy> cust::memory::DeviceCopy for FixedSizeBufferRef<'a, T> {}

#[cfg(all(not(target_os = "cuda"), feature = "gpu"))]
unsafe impl<'a, T: 'a + Copy> cust::memory::DeviceCopy for FixedSizeBufferMut<'a, T> {}

unsafe impl<'a, T: 'a + Copy + Sync> Sync for FixedSizeBufferRef<'a, T> {}
unsafe impl<'a, T: 'a + Copy + Send> Send for FixedSizeBufferRef<'a, T> {}
unsafe impl<'a, T: 'a + Copy + Sync> Sync for FixedSizeBufferMut<'a, T> {}
unsafe impl<'a, T: 'a + Copy + Send> Send for FixedSizeBufferMut<'a, T> {}

impl<'a, T: 'a + Copy> FixedSizeBufferRef<'a, T> {
    pub(crate) fn new(data: *const T, size: u64) -> Self {
        Self {
            data,
            size,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: 'a + Copy> FixedSizeBufferMut<'a, T> {
    pub(crate) fn new(data: *mut T, size: u64) -> Self {
        Self {
            data,
            size,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: 'a + Copy> Deref for FixedSizeBufferRef<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe {
            from_raw_parts(self.data, self.size as usize)
        }
    }
}

impl<'a, T: 'a + Copy> Deref for FixedSizeBufferMut<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe {
            from_raw_parts(self.data, self.size as usize)
        }
    }
}

impl<'a, T: 'a + Copy> DerefMut for FixedSizeBufferMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            from_raw_parts_mut(self.data, self.size as usize)
        }
    }
}

impl<'a, T: 'a + Copy> AsRef<[T]> for FixedSizeBufferRef<'a, T> {
    fn as_ref(&self) -> &[T] {
        unsafe {
            from_raw_parts(self.data, self.size as usize)
        }
    }
}

impl<'a, T: 'a + Copy> AsRef<[T]> for FixedSizeBufferMut<'a, T> {
    fn as_ref(&self) -> &[T] {
        unsafe {
            from_raw_parts(self.data, self.size as usize)
        }
    }
}

impl<'a, T: 'a + Copy> AsMut<[T]> for FixedSizeBufferMut<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe {
            from_raw_parts_mut(self.data, self.size as usize)
        }
    }
}

impl<'a, T: 'a + Copy> Borrow<[T]> for FixedSizeBufferRef<'a, T> {
    fn borrow(&self) -> &[T] {
        unsafe {
            from_raw_parts(self.data, self.size as usize)
        }
    }
}

impl<'a, T: 'a + Copy> Borrow<[T]> for FixedSizeBufferMut<'a, T> {
    fn borrow(&self) -> &[T] {
        unsafe {
            from_raw_parts(self.data, self.size as usize)
        }
    }
}

impl<'a, T: 'a + Copy> BorrowMut<[T]> for FixedSizeBufferMut<'a, T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        unsafe {
            from_raw_parts_mut(self.data, self.size as usize)
        }
    }
}

pub trait GetFixedSizeBufferRef<'a, T: Copy> {
    fn get_ref(&self) -> FixedSizeBufferRef<'a, T>;
}

pub trait GetFixedSizeBufferMut<'a, T: Copy> {
    fn get_mut(&mut self) -> FixedSizeBufferMut<'a, T>;
}

#[cfg(feature = "cpu")]
pub mod cpu {
    use super::*;

    #[derive(Debug)]
    pub struct FixedSizeCPUBuffer<'a, T: Copy> {
        pub(crate) buffer: Vec<T>,
        _marker: PhantomData<&'a T>,
    }

    impl<'a, T: Copy> FixedSizeCPUBuffer<'a, T> {
        pub fn clone<'b>(&self) -> FixedSizeCPUBuffer<'b, T> {
            FixedSizeCPUBuffer {
                buffer: self.buffer.clone(),
                _marker: PhantomData,
            }
        }
    }

    impl<'a, T: Copy> From<Vec<T>> for FixedSizeCPUBuffer<'a, T> {
        fn from(buffer: Vec<T>) -> Self {
            Self {
                buffer,
                _marker: PhantomData,
            }
        }
    }

    impl<'a, T: Copy> GetFixedSizeBufferRef<'a, T> for FixedSizeCPUBuffer<'a, T> {
        fn get_ref(&self) -> FixedSizeBufferRef<'a, T> {
            FixedSizeBufferRef::from(self)
        }
    }

    impl<'a, T: Copy> GetFixedSizeBufferMut<'a, T> for FixedSizeCPUBuffer<'a, T> {
        fn get_mut(&mut self) -> FixedSizeBufferMut<'a, T> {
            FixedSizeBufferMut::from(self)
        }
    }

    impl<'a, T: Copy> From<&FixedSizeCPUBuffer<'a, T>> for FixedSizeBufferRef<'a, T> {
        fn from(source: &FixedSizeCPUBuffer<'a, T>) -> Self {
            FixedSizeBufferRef::new(source.buffer.as_ptr(), source.buffer.len() as u64)
        }
    }

    impl<'a, T: Copy> From<&mut FixedSizeCPUBuffer<'a, T>> for FixedSizeBufferMut<'a, T> {
        fn from(source: &mut FixedSizeCPUBuffer<'a, T>) -> Self {
            FixedSizeBufferMut::new(source.buffer.as_mut_ptr(), source.buffer.len() as u64)
        }
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use cust::error::CudaResult;
    use cust::memory::{AsyncCopyDestination, CopyDestination, DeviceBuffer, DeviceCopy};
    use cust::stream::Stream;
    use super::*;

    #[derive(Debug)]
    pub struct FixedSizeGPUBuffer<'a, T: DeviceCopy> {
        buffer: DeviceBuffer<T>,
        _marker: PhantomData<&'a T>,
    }

    impl<'a, T: DeviceCopy> GetFixedSizeBufferRef<'a, T> for FixedSizeGPUBuffer<'a, T> {
        fn get_ref(&self) -> FixedSizeBufferRef<'a, T> {
            FixedSizeBufferRef::from(self)
        }
    }

    impl<'a, T: DeviceCopy> GetFixedSizeBufferMut<'a, T> for FixedSizeGPUBuffer<'a, T> {
        fn get_mut(&mut self) -> FixedSizeBufferMut<'a, T> {
            FixedSizeBufferMut::from(self)
        }
    }

    impl<'a, T: DeviceCopy> From<&FixedSizeGPUBuffer<'a, T>> for FixedSizeBufferRef<'a, T> {
        fn from(source: &FixedSizeGPUBuffer<'a, T>) -> Self {
            FixedSizeBufferRef::new(source.buffer.as_device_ptr().as_ptr(), source.buffer.len() as u64)
        }
    }

    impl<'a, T: DeviceCopy> From<&mut FixedSizeGPUBuffer<'a, T>> for FixedSizeBufferMut<'a, T> {
        fn from(source: &mut FixedSizeGPUBuffer<'a, T>) -> Self {
            FixedSizeBufferMut::new(source.buffer.as_device_ptr().as_mut_ptr(), source.buffer.len() as u64)
        }
    }

    impl<'a, T: DeviceCopy> FixedSizeGPUBuffer<'a, T> {
        pub fn copy_from_cpu(&mut self, c_buffer: &cpu::FixedSizeCPUBuffer<'_, T>) -> CudaResult<()> {
            self.buffer.copy_from(&c_buffer.buffer)
        }
        pub fn copy_to_cpu(&self, c_buffer: &mut cpu::FixedSizeCPUBuffer<'_, T>) -> CudaResult<()> {
            self.buffer.copy_to(&mut c_buffer.buffer)
        }

        pub unsafe fn async_copy_from_cpu(&mut self, c_buffer: &cpu::FixedSizeCPUBuffer<'_, T>, stream: &Stream) -> CudaResult<()> {
            self.buffer.async_copy_from(&c_buffer.buffer, stream)
        }
        pub unsafe fn async_copy_to_cpu(&self, c_buffer: &mut cpu::FixedSizeCPUBuffer<'_, T>, stream: &Stream) -> CudaResult<()> {
            self.buffer.async_copy_to(&mut c_buffer.buffer, stream)
        }

    }

    pub trait AsGPUBuffer {
        type HoldingType : DeviceCopy;
        fn as_gpu_buffer<'a>(&self) -> CudaResult<FixedSizeGPUBuffer<'a, Self::HoldingType>>;
        unsafe fn as_gpu_buffer_async<'a>(&self, stream: &Stream) -> CudaResult<FixedSizeGPUBuffer<'a, Self::HoldingType>>;
    }

    impl<'a, T: DeviceCopy> AsGPUBuffer for cpu::FixedSizeCPUBuffer<'a, T> {
        type HoldingType = T;

        fn as_gpu_buffer<'b>(&self) -> CudaResult<FixedSizeGPUBuffer<'b, Self::HoldingType>> {
            let buffer = DeviceBuffer::from_slice(&self.buffer)?;
            Ok(FixedSizeGPUBuffer {
                buffer,
                _marker: PhantomData,
            })
        }

        unsafe fn as_gpu_buffer_async<'b>(&self, stream: &Stream) -> CudaResult<FixedSizeGPUBuffer<'b, Self::HoldingType>> {
            let buffer = DeviceBuffer::from_slice_async(&self.buffer, stream)?;
            Ok(FixedSizeGPUBuffer {
                buffer,
                _marker: PhantomData,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cpu")]
    mod cpu {
        use super::*;
        use crate::cpu::*;

        struct TestInner<'a> {
            pub borrower: FixedSizeBufferRef<'a, u64>,
        }

        struct TestStruct<'a> {
            pub owner: FixedSizeCPUBuffer<'a, u64>,
            pub inner: TestInner<'a>,
        }

        impl <'a> Deref for TestStruct<'a> {
            type Target = TestInner<'a>;

            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl<'a> TestStruct<'a> {
            pub fn new(buffer: Vec<u64>) -> Self {
                let mut cpu_buffer = FixedSizeCPUBuffer::from(buffer);
                let borrower = cpu_buffer.get_ref();
                let inner = TestInner {
                    borrower,
                };
                TestStruct {
                    owner: cpu_buffer,
                    inner,
                }
            }
        }


        fn other_fn(s: &TestStruct<'_>) {
            assert_eq!(s.inner.borrower[2], 3);
        }

        fn second_fn(s: &mut TestStruct<'_>) {
            s.inner.borrower[2] = 4;
        }

        #[test]
        fn it_works() {
            let mut test = TestStruct::new(vec![1, 2, 3]);
            other_fn(&test);
            second_fn(&mut test);
            assert_eq!(test.inner.borrower[2], 4);
            assert_eq!(test.owner.buffer[2], 4);
        }
    }
}
