// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with linear mapping.
//!
//! See [`LinearMap`] for details on how to use it.

use crate::{
    paging::{
        deallocate, is_aligned, Attributes, MemoryRegion, PageTable, PhysicalAddress, Translation,
        VaRange, VirtualAddress, PAGE_SIZE,
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
    ///
    /// The `offset` must be a multiple of [`PAGE_SIZE`]; if not this will panic.
    pub fn new(offset: isize) -> Self {
        if !is_aligned(offset.unsigned_abs(), PAGE_SIZE) {
            panic!(
                "Invalid offset {}, must be a multiple of page size {}.",
                offset, PAGE_SIZE,
            );
        }
        Self { offset }
    }

    fn virtual_to_physical(&self, va: VirtualAddress) -> Result<PhysicalAddress, MapError> {
        if let Some(pa) = checked_add_to_unsigned(va.0 as isize, self.offset) {
            Ok(PhysicalAddress(pa))
        } else {
            Err(MapError::InvalidVirtualAddress(va))
        }
    }
}

impl Translation for LinearTranslation {
    fn allocate_table(&self) -> (NonNull<PageTable>, PhysicalAddress) {
        let table = PageTable::new();
        // Assume that the same linear mapping is used everywhere.
        let va = VirtualAddress(table.as_ptr() as usize);

        let pa = self.virtual_to_physical(va).expect(
            "Allocated subtable with virtual address which doesn't correspond to any physical address."
        );
        (table, pa)
    }

    unsafe fn deallocate_table(&self, page_table: NonNull<PageTable>) {
        deallocate(page_table);
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable> {
        let signed_pa = pa.0 as isize;
        if signed_pa < 0 {
            panic!("Invalid physical address {} for pagetable", pa);
        }
        if let Some(va) = signed_pa.checked_sub(self.offset) {
            if let Some(ptr) = NonNull::new(va as *mut PageTable) {
                ptr
            } else {
                panic!(
                    "Invalid physical address {} for pagetable (translated to virtual address 0)",
                    pa
                )
            }
        } else {
            panic!("Invalid physical address {} for pagetable", pa);
        }
    }
}

/// Adds two signed values, returning an unsigned value or `None` if it would overflow.
fn checked_add_to_unsigned(a: isize, b: isize) -> Option<usize> {
    a.checked_add(b)?.try_into().ok()
}

/// Manages a level 1 page table using linear mapping, where every virtual address is either
/// unmapped or mapped to an IPA with a fixed offset.
///
/// This assumes that the same linear mapping is used both for the page table being managed, and for
/// code that is managing it.
#[derive(Debug)]
pub struct LinearMap {
    mapping: Mapping<LinearTranslation>,
}

impl LinearMap {
    /// Creates a new identity-mapping page table with the given ASID, root level and offset, for
    /// use in the given TTBR.
    ///
    /// This will map any virtual address `va` which is added to the table to the physical address
    /// `va + offset`.
    ///
    /// The `offset` must be a multiple of [`PAGE_SIZE`]; if not this will panic.
    pub fn new(asid: usize, rootlevel: usize, offset: isize, va_range: VaRange) -> Self {
        Self {
            mapping: Mapping::new(LinearTranslation::new(offset), asid, rootlevel, va_range),
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
        paging::{Attributes, MemoryRegion, BITS_PER_LEVEL, PAGE_SIZE},
        MapError,
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;
    const GIB_512_S: isize = 512 * 1024 * 1024 * 1024;
    const GIB_512: usize = 512 * 1024 * 1024 * 1024;

    #[test]
    fn map_valid() {
        // A single byte at the start of the address space.
        let mut pagetable = LinearMap::new(1, 1, 4096, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, 1), Attributes::NORMAL),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = LinearMap::new(1, 1, 4096, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, PAGE_SIZE * 2), Attributes::NORMAL),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = LinearMap::new(1, 1, 4096, VaRange::Lower);
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

        // The entire valid address space. Use an offset that is a multiple of the level 2 block
        // size to avoid mapping everything as pages as that is really slow.
        const LEVEL_2_BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut pagetable = LinearMap::new(1, 1, LEVEL_2_BLOCK_SIZE as isize, VaRange::Lower);
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
        let mut pagetable = LinearMap::new(1, 1, -(PAGE_SIZE as isize), VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE + 1),
                Attributes::NORMAL
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = LinearMap::new(1, 1, -(PAGE_SIZE as isize), VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                Attributes::NORMAL
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = LinearMap::new(1, 1, -(PAGE_SIZE as isize), VaRange::Lower);
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

        // The entire valid address space. Use an offset that is a multiple of the level 2 block
        // size to avoid mapping everything as pages as that is really slow.
        const LEVEL_2_BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut pagetable = LinearMap::new(1, 1, -(LEVEL_2_BLOCK_SIZE as isize), VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(LEVEL_2_BLOCK_SIZE, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                Attributes::NORMAL
            ),
            Ok(())
        );
    }

    #[test]
    fn map_out_of_range() {
        let mut pagetable = LinearMap::new(1, 1, 4096, VaRange::Lower);

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
        let mut pagetable = LinearMap::new(1, 1, -4096, VaRange::Lower);

        // One byte, with an offset which would map it to a negative IPA.
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, 1), Attributes::NORMAL),
            Err(MapError::InvalidVirtualAddress(VirtualAddress(0)))
        );
    }

    #[test]
    fn physical_address_in_range_ttbr0() {
        let translation = LinearTranslation::new(4096);
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(8192)),
            NonNull::new(4096 as *mut PageTable).unwrap(),
        );
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(GIB_512 + 4096)),
            NonNull::new(GIB_512 as *mut PageTable).unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn physical_address_to_zero_ttbr0() {
        let translation = LinearTranslation::new(4096);
        translation.physical_to_virtual(PhysicalAddress(4096));
    }

    #[test]
    #[should_panic]
    fn physical_address_out_of_range_ttbr0() {
        let translation = LinearTranslation::new(4096);
        translation.physical_to_virtual(PhysicalAddress(-4096_isize as usize));
    }

    #[test]
    fn physical_address_in_range_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to one page above the bottom
        // of physical address space.
        let translation = LinearTranslation::new(GIB_512_S + 4096);
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(8192)),
            NonNull::new((4096 - GIB_512_S) as *mut PageTable).unwrap(),
        );
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(GIB_512)),
            NonNull::new(-4096_isize as *mut PageTable).unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn physical_address_to_zero_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to the bottom of physical
        // address space.
        let translation = LinearTranslation::new(GIB_512_S);
        translation.physical_to_virtual(PhysicalAddress(GIB_512));
    }

    #[test]
    #[should_panic]
    fn physical_address_out_of_range_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to the bottom of physical
        // address space.
        let translation = LinearTranslation::new(GIB_512_S);
        translation.physical_to_virtual(PhysicalAddress(-4096_isize as usize));
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

    #[test]
    fn virtual_address_range_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to the bottom of physical
        // address space.
        let translation = LinearTranslation::new(GIB_512_S);

        // The first page in the region covered by TTBR1.
        assert_eq!(
            translation.virtual_to_physical(VirtualAddress(0xffff_ff80_0000_0000)),
            Ok(PhysicalAddress(0))
        );
        // The last page in the region covered by TTBR1.
        assert_eq!(
            translation.virtual_to_physical(VirtualAddress(0xffff_ffff_ffff_f000)),
            Ok(PhysicalAddress(0x7f_ffff_f000))
        );
    }

    #[test]
    fn block_mapping() {
        // Test that block mapping is used when the PA is appropriately aligned...
        let mut pagetable = LinearMap::new(1, 1, 1 << 30, VaRange::Lower);
        pagetable
            .map_range(&MemoryRegion::new(0, 1 << 30), Attributes::NORMAL)
            .unwrap();
        assert_eq!(
            pagetable.mapping.root.mapping_level(VirtualAddress(0)),
            Some(1)
        );

        // ...but not when it is not.
        let mut pagetable = LinearMap::new(1, 1, 1 << 29, VaRange::Lower);
        pagetable
            .map_range(&MemoryRegion::new(0, 1 << 30), Attributes::NORMAL)
            .unwrap();
        assert_eq!(
            pagetable.mapping.root.mapping_level(VirtualAddress(0)),
            Some(2)
        );
    }
}
