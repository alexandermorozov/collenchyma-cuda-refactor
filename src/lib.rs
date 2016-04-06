extern crate collenchyma_refactor;

use std::any::Any;
use collenchyma_refactor::{Error, MemoryTransfer, Device};
use collenchyma_refactor::native::{NativeDevice, NativeMemory};


#[derive(Debug)]
pub struct CudaMemory {
    cuda_data: Vec<u8>
}

impl CudaMemory {
    // Toy impl just to test that it works

    /// Size in bytes
    pub fn size(&self) -> usize {
        self.cuda_data.len()
    }
    
    pub fn as_cuda_slice(&self) -> &[u8] {
        &self.cuda_data
    }

    pub fn as_mut_cuda_slice(&mut self) -> &mut [u8] {
        &mut self.cuda_data
    }
}


#[derive(PartialEq, Eq, Clone, Debug)]
pub struct CudaDevice {
    cuda_index: usize,
}

impl CudaDevice {
    pub fn new(cuda_index: usize) -> CudaDevice {
        CudaDevice {cuda_index: cuda_index}
    }
}

impl Device for CudaDevice {
    type M = CudaMemory;

    fn allocate_memory(_dev: &Self, size: usize) -> Result<Self::M, Error> {
        Ok(CudaMemory {
            cuda_data: vec![0; size]
        })
    }
}

impl MemoryTransfer for CudaDevice {
    /// "Cuda" can transfer to/from Native memory, but cannot transfer
    /// between two cuda devices directly in this implementation.
    fn transfer_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any)
                    -> Result<(), Error> {
        if let Some(_) = dst_device.downcast_ref::<NativeDevice>() {
            let my_mem = my_memory.downcast_ref::<CudaMemory>().unwrap();
            let mut dst_mem = dst_memory.downcast_mut::<NativeMemory>().unwrap();
            dst_mem.as_mut_slice().clone_from_slice(my_mem.as_cuda_slice());
            return Ok(());
        }

        Err(Error::NoMemoryTransferRoute)
    }

    fn transfer_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any)
                   -> Result<(), Error> {
        if let Some(_) = src_device.downcast_ref::<NativeDevice>() {
            let mut my_mem = my_memory.downcast_mut::<CudaMemory>().unwrap();
            let src_mem = src_memory.downcast_ref::<NativeMemory>().unwrap();
            my_mem.as_mut_cuda_slice().clone_from_slice(src_mem.as_slice());
            return Ok(());
        }

        Err(Error::NoMemoryTransferRoute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use collenchyma_refactor::SharedTensor;
    use collenchyma_refactor::native::{NativeDevice, NativeMemory};

    #[test]
    fn cuda_transfer_works() {
        let ndev = NativeDevice::new(0);
        let cdev = CudaDevice::new(0);

        let mut s1 = SharedTensor::new(vec![1,2,3]);
        for x in s1.write_only(&cdev).unwrap().mut_mem().as_mut_cuda_slice() {
            *x = 11;
        }
        assert_eq!(s1.read(&ndev).unwrap().mem().as_slice(),
                   [11, 11, 11, 11, 11, 11])
    }
}
