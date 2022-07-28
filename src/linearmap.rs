// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with linear mapping.
//!
//! See [`LinearMap`] for details on how to use it.

use crate::{
    paging::{
        deallocate, Attributes, MemoryRegion, PageTable, PhysicalAddress, Translation,
        VirtualAddress,
    },
    MapError, Mapping,
};
use core::ptr::NonNull;

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

impl LinearTranslation {
    fn virtual_to_physical(&self, va: VirtualAddress) -> Result<PhysicalAddress, MapError> {
        if let Some(pa) = checked_add_signed(va.0, self.offset) {
            Ok(PhysicalAddress(pa))
        } else {
            Err(MapError::InvalidVirtualAddress(va))
        }
    }
}

impl Translation for LinearTranslation {
    fn allocate_table(&self) -> (NonNull<PageTable>, PhysicalAddress) {
        let table = PageTable::new();
        let va = VirtualAddress::from(table.as_ptr());

        let pa = self.virtual_to_physical(va).expect(
            "Allocated subtable with virtual address which doesn't correspond to any physical address."
        );
        (table, pa)
    }

    unsafe fn deallocate_table(&self, page_table: NonNull<PageTable>) {
        deallocate(page_table);
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

/// Manages a level 1 page table using linear mapping, where every virtual address is either
/// unmapped or mapped to an IPA with a fixed offset.
#[derive(Debug)]
pub struct LinearMap {
    mapping: Mapping<LinearTranslation>,
}

impl LinearMap {
    /// Creates a new identity-mapping page table with the given ASID, root level and offset.
    ///
    /// This will map any virtual address `va` which is added to the table to the physical address
    /// `va + offset`.
    pub fn new(asid: usize, rootlevel: usize, offset: isize) -> Self {
        Self {
            mapping: Mapping::new(LinearTranslation::new(offset), asid, rootlevel),
        }
    }

    /// Activates the page table by setting `TTBR0_EL1` to point to it, and saves the previous value
    /// of `TTBR0_EL1` so that it may later be restored by [`deactivate`](Self::deactivate).
    ///
    /// Panics if a previous value of `TTBR0_EL1` is already saved and not yet used by a call to
    /// `deactivate`.
    #[cfg(target_arch = "aarch64")]
    pub fn activate(&mut self) {
        self.mapping.activate()
    }

    /// Deactivates the page table, by setting `TTBR0_EL1` back to the value it had before
    /// [`activate`](Self::activate) was called, and invalidating the TLB for this page table's
    /// configured ASID.
    ///
    /// Panics if there is no saved `TTRB0_EL1` value because `activate` has not previously been
    /// called.
    #[cfg(target_arch = "aarch64")]
    pub fn deactivate(&mut self) {
        self.mapping.deactivate()
    }

    /// Maps the given range of virtual addresses to the corresponding physical addresses with the
    /// given flags.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active.
    ///
    /// # Errors
    ///
    /// Returns [`MapError::InvalidVirtualAddress`] if adding the configured offset to any virtual
    /// address within the `range` would result in overflow.
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    pub fn map_range(&mut self, range: &MemoryRegion, flags: Attributes) -> Result<(), MapError> {
        let pa = self
            .mapping
            .root
            .translation()
            .virtual_to_physical(range.start())?;
        self.mapping.map_range(range, pa, flags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        paging::{Attributes, MemoryRegion, PAGE_SIZE},
        MapError,
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;

    #[test]
    fn map_valid() {
        // A single byte at the start of the address space.
        let mut pagetable = LinearMap::new(1, 1, 4096);
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, 1), Attributes::NORMAL),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = LinearMap::new(1, 1, 4096);
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, PAGE_SIZE * 2), Attributes::NORMAL),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = LinearMap::new(1, 1, 4096);
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
        let mut pagetable = LinearMap::new(1, 1, 4096);
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
        let mut pagetable = LinearMap::new(1, 1, -(PAGE_SIZE as isize));
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE + 1),
                Attributes::NORMAL
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = LinearMap::new(1, 1, -(PAGE_SIZE as isize));
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                Attributes::NORMAL
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = LinearMap::new(1, 1, -(PAGE_SIZE as isize));
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
        let mut pagetable = LinearMap::new(1, 1, -(PAGE_SIZE as isize));
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
        let mut pagetable = LinearMap::new(1, 1, 4096);

        // One byte, just past the edge of the valid range.
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1,
                ),
                Attributes::NORMAL
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );

        // From 0 to just past the valid range.
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1),
                Attributes::NORMAL
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );
    }

    #[test]
    fn map_invalid_offset() {
        let mut pagetable = LinearMap::new(1, 1, -4096);

        // One byte, with an offset which would map it to a negative IPA.
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, 1), Attributes::NORMAL),
            Err(MapError::InvalidVirtualAddress(VirtualAddress(0)))
        );
    }

    #[test]
    #[should_panic]
    fn physical_address_out_of_range() {
        let translation = LinearTranslation::new(4096);
        translation.physical_to_virtual(PhysicalAddress(1024));
    }

    #[test]
    fn virtual_address_out_of_range() {
        let translation = LinearTranslation::new(-4096);
        let va = VirtualAddress(1024);
        assert_eq!(
            translation.virtual_to_physical(va),
            Err(MapError::InvalidVirtualAddress(va))
        )
    }
}
