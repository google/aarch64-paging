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
    /// Constructs a new linear translation, which will map a virtual address `va` to the
    /// (intermediate) physical address `va + offset`.
    pub fn new(offset: isize) -> Self {
        Self { offset }
    }
}

impl Translation for LinearTranslation {
    fn virtual_to_physical(&self, va: VirtualAddress) -> PhysicalAddress {
        if let Some(pa) = checked_add_signed(va.0, self.offset) {
            PhysicalAddress(pa)
        } else {
            panic!("Attempt to map invalid virtual address {}", va)
        }
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> VirtualAddress {
        if let Some(va) = checked_add_signed(pa.0, -self.offset) {
            VirtualAddress(va)
        } else {
            panic!("Invalid physical address {}", pa);
        }
    }
}

// TODO: Use `usize::checked_add_signed` once it is stable
// (https://github.com/rust-lang/rust/issues/87840)
fn checked_add_signed(a: usize, b: isize) -> Option<usize> {
    if b >= 0 {
        a.checked_add(b as usize)
    } else {
        a.checked_sub(b.unsigned_abs())
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
    fn map_valid_negative_offset() {
        // A single byte which maps to IPA 0.
        let mut pagetable = LinearMap::new(LinearTranslation::new(-(PAGE_SIZE as isize)), 1, 1);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE + 1),
                Attributes::NORMAL
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = LinearMap::new(LinearTranslation::new(-(PAGE_SIZE as isize)), 1, 1);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                Attributes::NORMAL
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = LinearMap::new(LinearTranslation::new(-(PAGE_SIZE as isize)), 1, 1);
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
        let mut pagetable = LinearMap::new(LinearTranslation::new(-(PAGE_SIZE as isize)), 1, 1);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
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

    #[test]
    #[should_panic]
    fn physical_address_out_of_range() {
        let translation = LinearTranslation::new(4096);
        translation.physical_to_virtual(PhysicalAddress(1024));
    }

    #[test]
    #[should_panic]
    fn virtual_address_out_of_range() {
        let translation = LinearTranslation::new(-4096);
        translation.virtual_to_physical(VirtualAddress(1024));
    }
}
