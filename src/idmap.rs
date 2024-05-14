// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with identity mapping.
//!
//! See [`IdMap`] for details on how to use it.

use crate::{
    paging::{
        deallocate, Attributes, Constraints, Descriptor, MemoryRegion, PageTable, PhysicalAddress,
        Translation, TranslationRegime, VaRange, VirtualAddress,
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
/// ```no_run
/// use aarch64_paging::{
///     idmap::IdMap,
///     paging::{Attributes, MemoryRegion, TranslationRegime},
/// };
///
/// const ASID: usize = 1;
/// const ROOT_LEVEL: usize = 1;
///
/// // Create a new EL1 page table with identity mapping.
/// let mut idmap = IdMap::new(ASID, ROOT_LEVEL, TranslationRegime::El1);
/// // Map a 2 MiB region of memory as read-write.
/// idmap.map_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::VALID,
/// ).unwrap();
/// // SAFETY: Everything the program uses is within the 2 MiB region mapped above.
/// unsafe {
///     // Set `TTBR0_EL1` to activate the page table.
///     idmap.activate();
/// }
///
/// // Write something to the memory...
///
/// // SAFETY: The program will only use memory within the initially mapped region until `idmap` is
/// // reactivated below.
/// unsafe {
///     // Restore `TTBR0_EL1` to its earlier value while we modify the page table.
///     idmap.deactivate();
/// }
/// // Now change the mapping to read-only and executable.
/// idmap.map_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::READ_ONLY | Attributes::VALID,
/// ).unwrap();
/// // SAFETY: Everything the program will used is mapped in by this page table.
/// unsafe {
///     idmap.activate();
/// }
/// ```
#[derive(Debug)]
pub struct IdMap {
    mapping: Mapping<IdTranslation>,
}

impl IdMap {
    /// Creates a new identity-mapping page table with the given ASID and root level.
    pub fn new(asid: usize, rootlevel: usize, translation_regime: TranslationRegime) -> Self {
        Self {
            mapping: Mapping::new(
                IdTranslation,
                asid,
                rootlevel,
                translation_regime,
                VaRange::Lower,
            ),
        }
    }

    /// Activates the page table by setting `TTBR0_EL1` to point to it, and saves the previous value
    /// of `TTBR0_EL1` so that it may later be restored by [`deactivate`](Self::deactivate).
    ///
    /// Panics if a previous value of `TTBR0_EL1` is already saved and not yet used by a call to
    /// `deactivate`.
    ///
    /// In test builds or builds that do not target aarch64, the `TTBR0_EL1` access is omitted.
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

    /// Deactivates the page table, by setting `TTBR0_EL1` back to the value it had before
    /// [`activate`](Self::activate) was called, and invalidating the TLB for this page table's
    /// configured ASID.
    ///
    /// Panics if there is no saved `TTBR0_EL1` value because `activate` has not previously been
    /// called.
    ///
    /// In test builds or builds that do not target aarch64, the `TTBR0_EL1` access is omitted.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the previous page table which this is switching back to doesn't
    /// unmap any memory which the program is using.
    pub unsafe fn deactivate(&mut self) {
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
    ///
    /// Returns [`MapError::InvalidFlags`] if the `flags` argument has unsupported attributes set.
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation'] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    pub fn map_range(&mut self, range: &MemoryRegion, flags: Attributes) -> Result<(), MapError> {
        self.map_range_with_constraints(range, flags, Constraints::empty())
    }

    /// Maps the given range of virtual addresses to the identical physical addresses with the given
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
        let pa = IdTranslation::virtual_to_physical(range.start());
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
        MapError, VirtualAddress,
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;

    #[test]
    fn map_valid() {
        // A single byte at the start of the address space.
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, 1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }
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
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(PAGE_SIZE * 1023, PAGE_SIZE * 1025),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );

        // The entire valid address space.
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                Attributes::NORMAL | Attributes::VALID
            ),
            Ok(())
        );
    }

    #[test]
    fn map_break_before_make() {
        const BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        idmap
            .map_range_with_constraints(
                &MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE),
                Attributes::NORMAL | Attributes::VALID,
                Constraints::NO_BLOCK_MAPPINGS,
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }

        // Splitting a range is permitted if it was mapped down to pages
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(BLOCK_SIZE, BLOCK_SIZE + PAGE_SIZE),
                Attributes::NORMAL | Attributes::VALID,
            ),
            Ok(())
        );

        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        idmap
            .map_range(
                &MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE),
                Attributes::NORMAL | Attributes::VALID,
            )
            .ok();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }

        // Extending a range is fine even if there are block mappings
        // in the middle
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(BLOCK_SIZE - PAGE_SIZE, 2 * BLOCK_SIZE + PAGE_SIZE),
                Attributes::NORMAL | Attributes::VALID,
            ),
            Ok(())
        );

        // Splitting a range is not permitted
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(BLOCK_SIZE, BLOCK_SIZE + PAGE_SIZE),
                Attributes::NORMAL | Attributes::VALID,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );

        // Remapping a partially live range read-only is only permitted
        // if it does not require splitting
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, BLOCK_SIZE + PAGE_SIZE),
                Attributes::NORMAL | Attributes::VALID | Attributes::READ_ONLY,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, BLOCK_SIZE),
                Attributes::NORMAL | Attributes::VALID | Attributes::READ_ONLY,
            ),
            Ok(())
        );

        // Changing the memory type is not permitted
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, BLOCK_SIZE),
                Attributes::DEVICE_NGNRE | Attributes::VALID | Attributes::NON_GLOBAL,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                0, PAGE_SIZE
            )))
        );

        // Making a range invalid is only permitted if it does not require splitting
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(PAGE_SIZE, BLOCK_SIZE + PAGE_SIZE),
                Attributes::NORMAL,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(PAGE_SIZE, BLOCK_SIZE),
                Attributes::NORMAL,
            ),
            Ok(())
        );

        // Creating a new valid entry is always permitted
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, 2 * PAGE_SIZE),
                Attributes::NORMAL | Attributes::VALID,
            ),
            Ok(())
        );

        // Setting the non-global attribute is permitted
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                Attributes::NORMAL | Attributes::VALID | Attributes::NON_GLOBAL,
            ),
            Ok(())
        );

        // Removing the non-global attribute from a live mapping is not permitted
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                Attributes::NORMAL | Attributes::VALID,
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                0, PAGE_SIZE
            )))
        );

        // SAFETY: This doesn't actually deactivate the page table in tests, it just treats it as
        // inactive for the sake of BBM rules.
        unsafe {
            idmap.deactivate();
        }
        // Removing the non-global attribute from an inactive mapping is permitted
        assert_eq!(
            idmap.map_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                Attributes::NORMAL | Attributes::VALID,
            ),
            Ok(())
        );
    }

    #[test]
    fn map_out_of_range() {
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);

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
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        idmap
            .map_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                Attributes::NORMAL
                    | Attributes::NON_GLOBAL
                    | Attributes::READ_ONLY
                    | Attributes::VALID,
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }
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
        assert!(idmap
            .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry, level| {
                if level == 3 || !entry.is_table_or_page() {
                    entry.modify_flags(Attributes::SWFLAG_0, Attributes::NON_GLOBAL);
                }
                Ok(())
            })
            .is_err());
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
        let mut idmap = IdMap::new(1, 1, TranslationRegime::El1);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        unsafe {
            idmap.activate();
        }
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
