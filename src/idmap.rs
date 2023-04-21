// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with identity mapping.
//!
//! See [`IdMap`] for details on how to use it.

use crate::{
    paging::{
        deallocate, Attributes, MemoryRegion, PageTable, PhysicalAddress, PteUpdater, Translation,
        VaRange, VirtualAddress,
    },
    MapError, Mapping,
};
use core::ptr::NonNull;

/// Identity mapping, where every virtual address is either unmapped or mapped to the identical IPA.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IdTranslation;

impl IdTranslation {
    fn virtual_to_physical(va: VirtualAddress) -> PhysicalAddress {
        PhysicalAddress(va.0)
    }
}

impl Translation for IdTranslation {
    fn allocate_table(&self) -> (NonNull<PageTable>, PhysicalAddress) {
        let table = PageTable::new();

        // Physical address is the same as the virtual address because we are using identity mapping
        // everywhere.
        (table, PhysicalAddress(table.as_ptr() as usize))
    }

    unsafe fn deallocate_table(&self, page_table: NonNull<PageTable>) {
        deallocate(page_table);
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable> {
        NonNull::new(pa.0 as *mut PageTable).expect("Got physical address 0 for pagetable")
    }
}

/// Manages a level 1 page table using identity mapping, where every virtual address is either
/// unmapped or mapped to the identical IPA.
///
/// This assumes that identity mapping is used both for the page table being managed, and for code
/// that is managing it.
///
/// Mappings should be added with [`map_range`](Self::map_range) before calling
/// [`activate`](Self::activate) to start using the new page table. To make changes which may
/// require break-before-make semantics you must first call [`deactivate`](Self::deactivate) to
/// switch back to a previous static page table, and then `activate` again after making the desired
/// changes.
///
/// # Example
///
/// ```
/// use aarch64_paging::{
///     idmap::IdMap,
///     paging::{Attributes, MemoryRegion},
/// };
///
/// const ASID: usize = 1;
/// const ROOT_LEVEL: usize = 1;
///
/// // Create a new page table with identity mapping.
/// let mut idmap = IdMap::new(ASID, ROOT_LEVEL);
/// // Map a 2 MiB region of memory as read-write.
/// idmap.map_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::EXECUTE_NEVER | Attributes::VALID,
/// ).unwrap();
/// // Set `TTBR0_EL1` to activate the page table.
/// # #[cfg(target_arch = "aarch64")]
/// idmap.activate();
///
/// // Write something to the memory...
///
/// // Restore `TTBR0_EL1` to its earlier value while we modify the page table.
/// # #[cfg(target_arch = "aarch64")]
/// idmap.deactivate();
/// // Now change the mapping to read-only and executable.
/// idmap.map_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::READ_ONLY | Attributes::VALID,
/// ).unwrap();
/// # #[cfg(target_arch = "aarch64")]
/// idmap.activate();
/// ```
#[derive(Debug)]
pub struct IdMap {
    mapping: Mapping<IdTranslation>,
}

impl IdMap {
    /// Creates a new identity-mapping page table with the given ASID and root level.
    pub fn new(asid: usize, rootlevel: usize) -> Self {
        Self {
            mapping: Mapping::new(IdTranslation, asid, rootlevel, VaRange::Lower),
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

    /// Maps the given range of virtual addresses to the identical physical addresses with the given
    /// flags.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active. This function writes block and page entries, but only maps them if `flags`
    /// contains `Attributes::VALID`, otherwise the entries remain invalid.
    ///
    /// # Errors
    ///
    /// Returns [`MapError::RegionBackwards`] if the range is backwards.
    ///
    /// Returns [`MapError::AddressRange`] if the largest address in the `range` is greater than the
    /// largest virtual address covered by the page table given its root level.
    pub fn map_range(&mut self, range: &MemoryRegion, flags: Attributes) -> Result<(), MapError> {
        let pa = IdTranslation::virtual_to_physical(range.start());
        self.mapping.map_range(range, pa, flags)
    }

    /// Applies the provided updater function to a number of PTEs corresponding to a given memory range.
    ///
    /// The virtual address range passed to the updater function may be expanded compared to the
    /// `range` parameter, due to alignment to block boundaries.
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
    pub fn modify_range(&mut self, range: &MemoryRegion, f: &PteUpdater) -> Result<(), MapError> {
        self.mapping.modify_range(range, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        paging::{Attributes, MemoryRegion, PAGE_SIZE},
        MapError, VirtualAddress,
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;

    #[test]
    fn map_valid() {
        // A single byte at the start of the address space.
        let mut idmap = IdMap::new(1, 1);
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, 1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut idmap = IdMap::new(1, 1);
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut idmap = IdMap::new(1, 1);
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // Two pages, on the boundary between two subtables.
        let mut idmap = IdMap::new(1, 1);
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(PAGE_SIZE * 1023, PAGE_SIZE * 1025),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // The entire valid address space.
        let mut idmap = IdMap::new(1, 1);
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );
    }

    #[test]
    fn map_out_of_range() {
        let mut idmap = IdMap::new(1, 1);

        // One byte, just past the edge of the valid range.
        assert_eq!(
            idmap.map_range(
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
            idmap.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1,),
                Attributes::NORMAL | Attributes::VALID
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );
    }

    fn make_map() -> IdMap {
        let mut idmap = IdMap::new(1, 1);
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                Attributes::NORMAL
                    | Attributes::NON_GLOBAL
                    | Attributes::READ_ONLY
                    | Attributes::VALID,
            )
            .unwrap();
        idmap
    }

    #[test]
    fn update_backwards_range() {
        let mut idmap = make_map();
        assert!(idmap
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
        let mut idmap = make_map();
        idmap
            .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry, level| {
                if level == 3 || !entry.is_table_or_page() {
                    entry
                        .modify_flags(Attributes::SWFLAG_0, Attributes::from_bits(0usize).unwrap());
                }
                Ok(())
            })
            .unwrap();
        idmap
            .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|range, entry, level| {
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
        let mut idmap = IdMap::new(1, 1);
        idmap
            .map_range(
                &MemoryRegion::new(0, BLOCK_RANGE),
                Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::SWFLAG_0,
            )
            .unwrap();
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::VALID,
            )
            .unwrap();
        idmap
            .modify_range(
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
