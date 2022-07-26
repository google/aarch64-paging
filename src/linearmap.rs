// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with linear mapping.

use crate::{
    paging::{PhysicalAddress, Translation, VirtualAddress},
    Mapping,
};

/// Linear mapping, where every virtual address is either unmapped or mapped to an IPA with a fixed
/// offset.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LinearTranslation {
    /// The offset from a virtual address to the corresponding (intermediate) physical address.
    offset: isize,
}

impl LinearTranslation {
    pub fn new(offset: isize) -> Self {
        Self { offset }
    }
}

impl Translation for LinearTranslation {
    fn virtual_to_physical(&self, va: VirtualAddress) -> PhysicalAddress {
        PhysicalAddress(va.0.wrapping_add(self.offset as usize))
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> VirtualAddress {
        VirtualAddress(pa.0.wrapping_sub(self.offset as usize))
    }
}

pub type LinearMap = Mapping<LinearTranslation>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        paging::{Attributes, MemoryRegion, PAGE_SIZE},
        AddressRangeError,
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;

    #[test]
    fn map_valid() {
        // A single byte at the start of the address space.
        let mut pagetable = LinearMap::new(LinearTranslation::new(4096), 1, 1);
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, 1), Attributes::NORMAL),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = LinearMap::new(LinearTranslation::new(4096), 1, 1);
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, PAGE_SIZE * 2), Attributes::NORMAL),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = LinearMap::new(LinearTranslation::new(4096), 1, 1);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                Attributes::NORMAL
            ),
            Ok(())
        );

        // The entire valid address space.
        let mut pagetable = LinearMap::new(LinearTranslation::new(4096), 1, 1);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                Attributes::NORMAL
            ),
            Ok(())
        );
    }

    #[test]
    fn map_out_of_range() {
        let mut pagetable = LinearMap::new(LinearTranslation::new(4096), 1, 1);

        // One byte, just past the edge of the valid range.
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1,
                ),
                Attributes::NORMAL
            ),
            Err(AddressRangeError)
        );

        // From 0 to just past the valid range.
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1,),
                Attributes::NORMAL
            ),
            Err(AddressRangeError)
        );
    }
}
