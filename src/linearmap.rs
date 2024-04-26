// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with linear mapping.
//!
//! See [`LinearMap`] for details on how to use it.

use crate::{
    paging::{
        deallocate, is_aligned, Attributes, Constraints, Descriptor, ExceptionLevel, MemoryRegion,
        PageTable, PhysicalAddress, Translation, VaRange, VirtualAddress, PAGE_SIZE,
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
    pub fn new(
        asid: usize,
        rootlevel: usize,
        offset: isize,
        exception_level: ExceptionLevel,
        va_range: VaRange,
    ) -> Self {
        Self {
            mapping: Mapping::new(
                LinearTranslation::new(offset),
                asid,
                rootlevel,
                exception_level,
                va_range,
            ),
        }
    }

    /// Activates the page table by setting `TTBRn_EL1` to point to it, and saves the previous value
    /// of `TTBRn_EL1` so that it may later be restored by [`deactivate`](Self::deactivate).
    ///
    /// Panics if a previous value of `TTBRn_EL1` is already saved and not yet used by a call to
    /// `deactivate`.
    ///
    /// In test builds or builds that do not target aarch64, the `TTBRn_EL1` access is omitted.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the page table doesn't unmap any memory which the program is
    /// using, or introduce aliases which break Rust's aliasing rules. The page table must not be
    /// dropped as long as its mappings are required, as it will automatically be deactivated when
    /// it is dropped.
    pub unsafe fn activate(&mut self) {
        self.mapping.activate()
    }

    /// Deactivates the page table, by setting `TTBRn_EL1` back to the value it had before
    /// [`activate`](Self::activate) was called, and invalidating the TLB for this page table's
    /// configured ASID.
    ///
    /// Panics if there is no saved `TTBRn_EL1` value because `activate` has not previously been
    /// called.
    ///
    /// In test builds or builds that do not target aarch64, the `TTBRn_EL1` access is omitted.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the previous page table which this is switching back to doesn't
    /// unmap any memory which the program is using.
    pub unsafe fn deactivate(&mut self) {
        self.mapping.deactivate()
    }

    /// Maps the given range of virtual addresses to the corresponding physical addresses with the
    /// given flags.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active. This function writes block and page entries, but only maps them if `flags`
    /// contains `Attributes::VALID`, otherwise the entries remain invalid.
    ///
    /// # Errors
    ///
    /// Returns [`MapError::InvalidVirtualAddress`] if adding the configured offset to any virtual
    /// address within the `range` would result in overflow.
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    ///
    /// Returns [`MapError::InvalidFlags`] if the `flags` argument has unsupported attributes set.
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation'] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    pub fn map_range(&mut self, range: &MemoryRegion, flags: Attributes) -> Result<(), MapError> {
        self.map_range_with_constraints(range, flags, Constraints::empty())
    }

    /// Maps the given range of virtual addresses to the corresponding physical addresses with the
    /// given flags, taking the given constraints into account.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active. This function writes block and page entries, but only maps them if `flags`
    /// contains `Attributes::VALID`, otherwise the entries remain invalid.
    ///
    /// # Errors
    ///
    /// Returns [`MapError::InvalidVirtualAddress`] if adding the configured offset to any virtual
    /// address within the `range` would result in overflow.
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    ///
    /// Returns [`MapError::InvalidFlags`] if the `flags` argument has unsupported attributes set.
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation'] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    pub fn map_range_with_constraints(
        &mut self,
        range: &MemoryRegion,
        flags: Attributes,
        constraints: Constraints,
    ) -> Result<(), MapError> {
        let pa = self
            .mapping
            .root
            .translation()
            .virtual_to_physical(range.start())?;
        self.mapping.map_range(range, pa, flags, constraints)
    }

    /// Applies the provided updater function to the page table descriptors covering a given
    /// memory range.
    ///
    /// This may involve splitting block entries if the provided range is not currently mapped
    /// down to its precise boundaries. For visiting all the descriptors covering a memory range
    /// without potential splitting (and no descriptor updates), use
    /// [`walk_range`](Self::walk_range) instead.
    ///
    /// The updater function receives the following arguments:
    ///
    /// - The virtual address range mapped by each page table descriptor. A new descriptor will
    ///   have been allocated before the invocation of the updater function if a page table split
    ///   was needed.
    /// - A mutable reference to the page table descriptor that permits modifications.
    /// - The level of a translation table the descriptor belongs to.
    ///
    /// The updater function should return:
    ///
    /// - `Ok` to continue updating the remaining entries.
    /// - `Err` to signal an error and stop updating the remaining entries.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active.
    ///
    /// # Errors
    ///
    /// Returns [`MapError::PteUpdateFault`] if the updater function returns an error.
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation'] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    pub fn modify_range<F>(&mut self, range: &MemoryRegion, f: &F) -> Result<(), MapError>
    where
        F: Fn(&MemoryRegion, &mut Descriptor, usize) -> Result<(), ()> + ?Sized,
    {
        self.mapping.modify_range(range, f)
    }

    /// Applies the provided callback function to the page table descriptors covering a given
    /// memory range.
    ///
    /// The callback function receives the following arguments:
    ///
    /// - The range covered by the current step in the walk. This is always a subrange of `range`
    ///   even when the descriptor covers a region that exceeds it.
    /// - The page table descriptor itself.
    /// - The level of a translation table the descriptor belongs to.
    ///
    /// The callback function should return:
    ///
    /// - `Ok` to continue visiting the remaining entries.
    /// - `Err` to signal an error and stop visiting the remaining entries.
    ///
    /// # Errors
    ///
    /// Returns [`MapError::PteUpdateFault`] if the callback function returns an error.
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    pub fn walk_range<F>(&self, range: &MemoryRegion, f: &mut F) -> Result<(), MapError>
    where
        F: FnMut(&MemoryRegion, &Descriptor, usize) -> Result<(), ()>,
    {
        self.mapping.walk_range(range, f)
    }

    /// Returns the physical address of the root table.
    ///
    /// This may be used to activate the page table by setting the appropriate TTBRn_ELx if you wish
    /// to do so yourself rather than by calling [`activate`](Self::activate). Make sure to call
    /// [`mark_active`](Self::mark_active) after doing so.
    pub fn root_address(&self) -> PhysicalAddress {
        self.mapping.root_address()
    }

    /// Marks the page table as active.
    ///
    /// This should be called if the page table is manually activated by calling
    /// [`root_address`](Self::root_address) and setting some TTBR with it. This will cause
    /// [`map_range`](Self::map_range) and [`modify_range`](Self::modify_range) to perform extra
    /// checks to avoid violating break-before-make requirements.
    ///
    /// It is called automatically by [`activate`](Self::activate).
    pub fn mark_active(&mut self, previous_ttbr: usize) {
        self.mapping.mark_active(previous_ttbr);
    }

    /// Marks the page table as inactive.
    ///
    /// This may be called after manually disabling the use of the page table, such as by setting
    /// the relevant TTBR to a different address.
    ///
    /// It is called automatically by [`deactivate`](Self::deactivate).
    pub fn mark_inactive(&mut self) {
        self.mapping.mark_inactive();
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
        let mut pagetable = LinearMap::new(1, 1, 4096, ExceptionLevel::El1, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, 1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = LinearMap::new(1, 1, 4096, ExceptionLevel::El1, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = LinearMap::new(1, 1, 4096, ExceptionLevel::El1, VaRange::Lower);
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // The entire valid address space. Use an offset that is a multiple of the level 2 block
        // size to avoid mapping everything as pages as that is really slow.
        const LEVEL_2_BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut pagetable = LinearMap::new(
            1,
            1,
            LEVEL_2_BLOCK_SIZE as isize,
            ExceptionLevel::El1,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );
    }

    #[test]
    fn map_valid_negative_offset() {
        // A single byte which maps to IPA 0.
        let mut pagetable = LinearMap::new(
            1,
            1,
            -(PAGE_SIZE as isize),
            ExceptionLevel::El1,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE + 1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = LinearMap::new(
            1,
            1,
            -(PAGE_SIZE as isize),
            ExceptionLevel::El1,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = LinearMap::new(
            1,
            1,
            -(PAGE_SIZE as isize),
            ExceptionLevel::El1,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // The entire valid address space. Use an offset that is a multiple of the level 2 block
        // size to avoid mapping everything as pages as that is really slow.
        const LEVEL_2_BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut pagetable = LinearMap::new(
            1,
            1,
            -(LEVEL_2_BLOCK_SIZE as isize),
            ExceptionLevel::El1,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(LEVEL_2_BLOCK_SIZE, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );
    }

    #[test]
    fn map_out_of_range() {
        let mut pagetable = LinearMap::new(1, 1, 4096, ExceptionLevel::El1, VaRange::Lower);

        // One byte, just past the edge of the valid range.
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1,
                ),
                Attributes::NORMAL | Attributes::VALID
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );

        // From 0 to just past the valid range.
        assert_eq!(
            pagetable.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );
    }

    #[test]
    fn map_invalid_offset() {
        let mut pagetable = LinearMap::new(1, 1, -4096, ExceptionLevel::El1, VaRange::Lower);

        // One byte, with an offset which would map it to a negative IPA.
        assert_eq!(
            pagetable.map_range(&MemoryRegion::new(0, 1), Attributes::NORMAL,),
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
        let mut pagetable = LinearMap::new(1, 1, 1 << 30, ExceptionLevel::El1, VaRange::Lower);
        pagetable
            .map_range(
                &MemoryRegion::new(0, 1 << 30),
                Attributes::NORMAL | Attributes::VALID,
            )
            .unwrap();
        assert_eq!(
            pagetable.mapping.root.mapping_level(VirtualAddress(0)),
            Some(1)
        );

        // ...but not when it is not.
        let mut pagetable = LinearMap::new(1, 1, 1 << 29, ExceptionLevel::El1, VaRange::Lower);
        pagetable
            .map_range(
                &MemoryRegion::new(0, 1 << 30),
                Attributes::NORMAL | Attributes::VALID,
            )
            .unwrap();
        assert_eq!(
            pagetable.mapping.root.mapping_level(VirtualAddress(0)),
            Some(2)
        );
    }

    fn make_map() -> LinearMap {
        let mut lmap = LinearMap::new(1, 1, 4096, ExceptionLevel::El1, VaRange::Lower);
        // Mapping VA range 0x0 - 0x2000 to PA range 0x1000 - 0x3000
        lmap.map_range(&MemoryRegion::new(0, PAGE_SIZE * 2), Attributes::NORMAL)
            .unwrap();
        lmap
    }

    #[test]
    fn update_backwards_range() {
        let mut lmap = make_map();
        assert!(lmap
            .modify_range(
                &MemoryRegion::new(PAGE_SIZE * 2, 1),
                &|_range, entry, _level| {
                    entry
                        .modify_flags(Attributes::SWFLAG_0, Attributes::from_bits(0usize).unwrap());
                    Ok(())
                },
            )
            .is_err());
    }

    #[test]
    fn update_range() {
        let mut lmap = make_map();
        lmap.modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry, level| {
            if level == 3 || !entry.is_table_or_page() {
                entry.modify_flags(Attributes::SWFLAG_0, Attributes::from_bits(0usize).unwrap());
            }
            Ok(())
        })
        .unwrap();
        lmap.modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|range, entry, level| {
            if level == 3 || !entry.is_table_or_page() {
                assert!(entry.flags().unwrap().contains(Attributes::SWFLAG_0));
                assert_eq!(range.end() - range.start(), PAGE_SIZE);
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn breakup_invalid_block() {
        const BLOCK_RANGE: usize = 0x200000;

        let mut lmap = LinearMap::new(1, 1, 0x1000, ExceptionLevel::El1, VaRange::Lower);
        lmap.map_range(
            &MemoryRegion::new(0, BLOCK_RANGE),
            Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::SWFLAG_0,
        )
        .unwrap();
        lmap.map_range(
            &MemoryRegion::new(0, PAGE_SIZE),
            Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::VALID,
        )
        .unwrap();
        lmap.modify_range(
            &MemoryRegion::new(0, BLOCK_RANGE),
            &|range, entry, level| {
                if level == 3 {
                    let has_swflag = entry.flags().unwrap().contains(Attributes::SWFLAG_0);
                    let is_first_page = range.start().0 == 0usize;
                    assert!(has_swflag != is_first_page);
                }
                Ok(())
            },
        )
        .unwrap();
    }
}
