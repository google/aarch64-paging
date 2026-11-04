// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with identity mapping.
//!
//! See [`IdMap`] for details on how to use it.

use crate::{
    MapError, Mapping,
    descriptor::{PagingAttributes, PhysicalAddress, VirtualAddress},
    paging::{FixedTranslation, PageTable, Translation, TranslationRegime, deallocate},
};
use core::marker::PhantomData;
use core::ptr::NonNull;

/// Identity mapping, where every virtual address is either unmapped or mapped to the identical IPA.
///
/// Page tables using this translation always cover the lower VA range, so there is no way to
/// select the VA range.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IdTranslation<A: PagingAttributes> {
    _phantom: PhantomData<A>,
}

impl<A: PagingAttributes> Default for IdTranslation<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: PagingAttributes> IdTranslation<A> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<A: PagingAttributes> Translation<A> for IdTranslation<A> {
    fn allocate_table(&mut self) -> (NonNull<PageTable<A>>, PhysicalAddress) {
        let table = PageTable::new();

        // Physical address is the same as the virtual address because we are using identity mapping
        // everywhere.
        (table, PhysicalAddress(table.as_ptr() as usize))
    }

    unsafe fn deallocate_table(&mut self, page_table: NonNull<PageTable<A>>) {
        // SAFETY: Our caller promises that the memory was allocated by `allocate_table` on this
        // `IdTranslation` and not yet deallocated. `allocate_table` used the global allocator and
        // appropriate layout by calling `PageTable::new()`.
        unsafe {
            deallocate(page_table);
        }
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable<A>> {
        NonNull::new(pa.0 as *mut PageTable<A>).expect("Got physical address 0 for pagetable")
    }
}

impl<A: PagingAttributes> FixedTranslation for IdTranslation<A> {
    fn virtual_to_physical(&self, va: VirtualAddress) -> Result<PhysicalAddress, MapError> {
        Ok(PhysicalAddress(va.0))
    }
}

/// Manages a level 1 page table using identity mapping, where every virtual address is either
/// unmapped or mapped to the identical IPA.
///
/// This assumes that identity mapping is used both for the page table being managed, and for code
/// that is managing it.
///
/// Mappings should be added with [`map_fixed_range`](Self::map_fixed_range) before calling
/// [`activate`](Self::activate) to start using the new page table. To make changes which may
/// require break-before-make semantics you must first call [`deactivate`](Self::deactivate) to
/// switch back to a previous static page table, and then `activate` again after making the desired
/// changes.
///
/// # Example
///
/// ```no_run
/// use aarch64_paging::{
///     Mapping,
///     descriptor::El1Attributes,
///     idmap::IdTranslation,
///     paging::{Constraints, El1And0, MemoryRegion},
/// };
///
/// const ASID: usize = 1;
/// const ROOT_LEVEL: usize = 1;
/// const NORMAL_CACHEABLE: El1Attributes = El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE);
///
/// // Create a new EL1 page table with identity mapping, which implies the lower VA range.
/// let mut idmap = Mapping::new(IdTranslation::new(), ROOT_LEVEL, El1And0::with_asid(ASID));
/// // Map a 2 MiB region of memory as read-write.
/// idmap.map_fixed_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::VALID | El1Attributes::ACCESSED,
///     Constraints::empty(),
/// ).unwrap();
/// // SAFETY: Everything the program uses is within the 2 MiB region mapped above.
/// let ttbr = unsafe {
///     // Set `TTBR0_EL1` to activate the page table.
///     idmap.activate()
/// };
///
/// // Write something to the memory...
///
/// // SAFETY: The program will only use memory within the initially mapped region until `idmap` is
/// // reactivated below.
/// unsafe {
///     // Restore `TTBR0_EL1` to its earlier value while we modify the page table.
///     idmap.deactivate(ttbr);
/// }
/// // Now change the mapping to read-only and executable.
/// idmap.map_fixed_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::READ_ONLY | El1Attributes::VALID
///     | El1Attributes::ACCESSED,
///     Constraints::empty(),
/// ).unwrap();
/// // SAFETY: Everything the program will used is mapped in by this page table.
/// unsafe {
///     idmap.activate();
/// }
/// ```
pub type IdMap<R> = Mapping<IdTranslation<<R as TranslationRegime>::Attributes>, R>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MapError, VirtualAddress,
        descriptor::{El1Attributes, UpdatableDescriptor},
        paging::{BITS_PER_LEVEL, Constraints, El1And0, MemoryRegion, PAGE_SIZE},
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;
    const DEVICE_NGNRE: El1Attributes = El1Attributes::ATTRIBUTE_INDEX_0;
    const NORMAL_CACHEABLE: El1Attributes =
        El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE);

    #[test]
    fn map_valid() {
        // A single byte at the start of the address space.
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }

        // Two pages at the start of the address space.
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }

        // A single byte at the end of the address space.
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }

        // Two pages, on the boundary between two subtables.
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(PAGE_SIZE * 1023, PAGE_SIZE * 1025),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }

        // The entire valid address space.
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn map_break_before_make() {
        const BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        idmap
            .map_fixed_range(
                &MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::NO_BLOCK_MAPPINGS,
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Splitting a range is permitted if it was mapped down to pages
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(BLOCK_SIZE, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );
        unsafe {
            idmap.deactivate(ttbr);
        }

        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        idmap
            .map_fixed_range(
                &MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .ok();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Extending a range is fine even if there are block mappings
        // in the middle
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(BLOCK_SIZE - PAGE_SIZE, 2 * BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // Remapping a region that intersects a block mapping is permitted if it does not result in
        // a split.
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(BLOCK_SIZE, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // Splitting a range is not permitted
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(BLOCK_SIZE, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::READ_ONLY,
                Constraints::empty(),
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );

        // Partially remapping a live range read-only is only permitted
        // if it does not require splitting
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::READ_ONLY,
                Constraints::empty(),
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, BLOCK_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::READ_ONLY,
                Constraints::empty(),
            ),
            Ok(())
        );

        // Changing the memory type is not permitted
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, BLOCK_SIZE),
                DEVICE_NGNRE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::NON_GLOBAL,
                Constraints::empty(),
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                0, PAGE_SIZE
            )))
        );

        // Making a range invalid is only permitted if it does not require splitting
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(PAGE_SIZE, BLOCK_SIZE + PAGE_SIZE),
                NORMAL_CACHEABLE,
                Constraints::empty(),
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE,
                BLOCK_SIZE + PAGE_SIZE
            )))
        );
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(PAGE_SIZE, BLOCK_SIZE),
                NORMAL_CACHEABLE,
                Constraints::empty(),
            ),
            Ok(())
        );

        // Creating a new valid entry is always permitted
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, 2 * PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // Setting the non-global attribute is permitted
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED
                    | El1Attributes::NON_GLOBAL,
                Constraints::empty(),
            ),
            Ok(())
        );

        // Removing the non-global attribute from a live mapping is not permitted
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                0, PAGE_SIZE
            )))
        );

        // SAFETY: This doesn't actually deactivate the page table in tests, it just treats it as
        // inactive for the sake of BBM rules.
        unsafe {
            idmap.deactivate(ttbr);
        }
        // Removing the non-global attribute from an inactive mapping is permitted
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );
    }

    #[test]
    fn map_out_of_range() {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));

        // One byte, just past the edge of the valid range.
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1,
                ),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );

        // From 0 to just past the valid range.
        assert_eq!(
            idmap.map_fixed_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1 + 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Err(MapError::AddressRange(VirtualAddress(
                MAX_ADDRESS_FOR_ROOT_LEVEL_1 + PAGE_SIZE
            )))
        );
    }

    #[test]
    fn unmap_live_block_mapping_with_no_block_mappings() {
        const BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let block = MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE);
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        idmap
            .map_fixed_range(
                &block,
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules and the TLB emulation.
        let ttbr = unsafe { idmap.activate() };
        // Unmapping the entire block mapping must not split it first, which would violate
        // break-before-make, even though the constraints don't permit block mappings.
        idmap
            .map_fixed_range(
                &block,
                El1Attributes::empty(),
                Constraints::NO_BLOCK_MAPPINGS,
            )
            .unwrap();
        // SAFETY: This doesn't actually deactivate the page table in tests, it just treats it as
        // inactive for the sake of BBM rules and the TLB emulation.
        unsafe { idmap.deactivate(ttbr) };
        idmap
            .walk_range(&block, &mut |_, descriptor, level| {
                assert_eq!((level, descriptor.bits()), (2, 0));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn split_live_block_mapping() -> () {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE << BITS_PER_LEVEL),
                NORMAL_CACHEABLE
                    | El1Attributes::NON_GLOBAL
                    | El1Attributes::READ_ONLY
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::NON_GLOBAL
                    | El1Attributes::READ_ONLY
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        let r = idmap.map_fixed_range(
            &MemoryRegion::new(PAGE_SIZE, 2 * PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
            Constraints::empty(),
        );
        unsafe { idmap.deactivate(ttbr) };
        r.unwrap();
    }

    #[test]
    fn modify_live_mapping_failing_halfway() {
        let range = MemoryRegion::new(0, 2 * PAGE_SIZE);
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        idmap
            .map_fixed_range(
                &range,
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules and the TLB emulation.
        let ttbr = unsafe { idmap.activate() };

        // modify_range() calls the updater for both pages to check for break-before-make
        // violations first, so this one only fails after making the first page read-only.
        let calls = core::cell::Cell::new(0);
        let result = idmap.modify_range(&range, &|_, d| {
            calls.set(calls.get() + 1);
            if calls.get() > 3 {
                return Err(());
            }
            d.modify_flags(El1Attributes::READ_ONLY, El1Attributes::empty())
        });
        assert!(
            matches!(result, Err(MapError::PteUpdateFault(_))),
            "{result:?}"
        );
        let mut read_only = alloc::vec::Vec::new();
        idmap
            .walk_range(&range, &mut |_, d, _| {
                read_only.push(d.flags().contains(El1Attributes::READ_ONLY));
                Ok(())
            })
            .unwrap();
        assert_eq!(read_only, [true, false]);
        // The TLB must not hold a stale translation for the first page, which modify_range() checks
        // in test builds.
        idmap.modify_range(&range, &|_, _| Ok(())).unwrap();

        // SAFETY: This doesn't actually deactivate the page table in tests, it just treats it as
        // inactive for the sake of BBM rules and the TLB emulation.
        unsafe { idmap.deactivate(ttbr) };
    }

    #[test]
    fn modify_range_partially_covering_block() {
        const BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mapped = MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE);
        let unmapped = MemoryRegion::new(2 * BLOCK_SIZE, 3 * BLOCK_SIZE);
        let both = MemoryRegion::new(BLOCK_SIZE, 3 * BLOCK_SIZE);
        // Partially covers both the block mapping and the unmapped block after it.
        let range = MemoryRegion::new(BLOCK_SIZE + PAGE_SIZE, 2 * BLOCK_SIZE + PAGE_SIZE);
        let flags = NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED;
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        idmap
            .map_fixed_range(&mapped, flags, Constraints::empty())
            .unwrap();
        let set_on_valid = |set: El1Attributes| {
            move |_: &MemoryRegion, d: &mut UpdatableDescriptor<El1Attributes>| {
                if d.flags().contains(El1Attributes::VALID) {
                    d.modify_flags(set, El1Attributes::empty())?;
                }
                Ok(())
            }
        };
        let walk = |idmap: &IdMap<El1And0>, region: &MemoryRegion| {
            let mut descriptors = alloc::vec::Vec::new();
            idmap
                .walk_range(region, &mut |mr, d, level| {
                    descriptors.push((mr.clone(), level, d.flags()));
                    Ok(())
                })
                .unwrap();
            descriptors
        };
        let unsplit = [
            (mapped.clone(), 2, flags),
            (unmapped.clone(), 2, El1Attributes::empty()),
        ];

        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        // Updates that don't modify the block mapping must not split it, as that would violate
        // break-before-make.
        let accessed = set_on_valid(El1Attributes::ACCESSED);
        idmap.modify_range(&range, &accessed).unwrap();
        assert_eq!(walk(&idmap, &both), unsplit);
        // Updates that do modify it are not permitted.
        let swflag = set_on_valid(El1Attributes::SWFLAG_0);
        assert_eq!(
            idmap.modify_range(&range, &swflag),
            Err(MapError::BreakBeforeMakeViolation(MemoryRegion::new(
                BLOCK_SIZE + PAGE_SIZE,
                2 * BLOCK_SIZE
            )))
        );
        assert_eq!(walk(&idmap, &both), unsplit);
        // SAFETY: This doesn't actually deactivate the page table in tests, it just treats it as
        // inactive for the sake of BBM rules.
        unsafe { idmap.deactivate(ttbr) };

        // While the page table is inactive, updates that don't modify a block mapping still leave
        // it alone, and so do updates that fail for it, but updates that do modify it split it.
        idmap.modify_range(&range, &accessed).unwrap();
        assert_eq!(walk(&idmap, &both), unsplit);
        assert_eq!(
            idmap.modify_range(&range, &|_, _| Err(())),
            Err(MapError::PteUpdateFault(BLOCK_SIZE | flags.bits()))
        );
        assert_eq!(walk(&idmap, &both), unsplit);
        idmap.modify_range(&range, &swflag).unwrap();
        let pages = walk(&idmap, &mapped);
        assert_eq!(pages.len(), 1 << BITS_PER_LEVEL);
        for (i, (mr, level, page_flags)) in pages.into_iter().enumerate() {
            let start = BLOCK_SIZE + i * PAGE_SIZE;
            let mut expected = flags | El1Attributes::TABLE_OR_PAGE;
            if i > 0 {
                expected |= El1Attributes::SWFLAG_0;
            }
            assert_eq!(
                (mr, level, page_flags),
                (MemoryRegion::new(start, start + PAGE_SIZE), 3, expected)
            );
        }
        // The unmapped block is left alone, as the update doesn't modify invalid descriptors.
        assert_eq!(walk(&idmap, &unmapped), unsplit[1..]);
    }

    fn make_map() -> (IdMap<El1And0>, usize) {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                NORMAL_CACHEABLE
                    | El1Attributes::NON_GLOBAL
                    | El1Attributes::READ_ONLY
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        (idmap, ttbr)
    }

    #[test]
    fn update_backwards_range() {
        let (mut idmap, ttbr) = make_map();
        assert!(
            idmap
                .modify_range(&MemoryRegion::new(PAGE_SIZE * 2, 1), &|_range, entry| {
                    entry.modify_flags(
                        El1Attributes::SWFLAG_0,
                        El1Attributes::from_bits(0usize).unwrap(),
                    )
                },)
                .is_err()
        );

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn update_range() {
        let (mut idmap, ttbr) = make_map();
        assert!(
            idmap
                .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry| {
                    if !entry.is_table() {
                        entry.modify_flags(El1Attributes::SWFLAG_0, El1Attributes::NON_GLOBAL)?;
                    }
                    Ok(())
                })
                .is_err()
        );
        idmap
            .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry| {
                if !entry.is_table() {
                    entry.modify_flags(
                        El1Attributes::SWFLAG_0,
                        El1Attributes::from_bits(0usize).unwrap(),
                    )?;
                }
                Ok(())
            })
            .unwrap();
        idmap
            .modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|range, entry| {
                if !entry.is_table() {
                    assert!(entry.flags().contains(El1Attributes::SWFLAG_0));
                    assert_eq!(range.end() - range.start(), PAGE_SIZE);
                }
                Ok(())
            })
            .unwrap();
        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn breakup_invalid_block() {
        const BLOCK_RANGE: usize = 0x200000;
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, BLOCK_RANGE),
                NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::SWFLAG_0,
                Constraints::empty(),
            )
            .unwrap();
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE
                    | El1Attributes::NON_GLOBAL
                    | El1Attributes::VALID
                    | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        idmap
            .modify_range(&MemoryRegion::new(0, BLOCK_RANGE), &|range, entry| {
                if entry.level() == 3 {
                    let has_swflag = entry.flags().contains(El1Attributes::SWFLAG_0);
                    let is_first_page = range.start().0 == 0usize;
                    assert!(has_swflag != is_first_page);
                }
                Ok(())
            })
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn unmap_subtable() {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Map one page, which will cause subtables to be split out.
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // Unmap the whole table's worth of address space.
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE * 512 * 512),
                El1Attributes::empty(),
                Constraints::empty(),
            )
            .unwrap();
        // All entries in the top-level table should be 0.
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn unmap_subtable_higher() {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        const ROOT_GRANULARITY: usize = PAGE_SIZE * 512 * 512;
        // Map one page in the the second entry of the root table.
        idmap
            .map_fixed_range(
                &MemoryRegion::new(ROOT_GRANULARITY, ROOT_GRANULARITY + PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // Unmap the second entry of the root table.
        idmap
            .map_fixed_range(
                &MemoryRegion::new(ROOT_GRANULARITY, ROOT_GRANULARITY * 2),
                El1Attributes::empty(),
                Constraints::empty(),
            )
            .unwrap();
        // All entries in the top-level table should be 0.
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn compact() {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Map two pages, which will cause subtables to be split out.
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // Unmap the pages again.
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                El1Attributes::empty(),
                Constraints::empty(),
            )
            .unwrap();
        // Compact to remove the subtables.
        idmap.compact_subtables();
        // All entries in the top-level table should be 0.
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn compact_blocks() {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Map two blocks at level 2, which will cause subtables to be split out.
        const BLOCK_SIZE: usize = PAGE_SIZE * 512;
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, BLOCK_SIZE * 2),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // Unmap the blocks again.
        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, BLOCK_SIZE * 2),
                El1Attributes::empty(),
                Constraints::empty(),
            )
            .unwrap();
        // Compact to remove the subtables.
        idmap.compact_subtables();
        // All entries in the top-level table should be 0.
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    /// When an unmapped entry is split into a table, all entries should be zero.
    #[test]
    fn split_table_zero() {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));

        idmap
            .map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        idmap
            .walk_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 20),
                &mut |_, descriptor, _| {
                    assert!(!descriptor.is_valid());
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(descriptor.flags(), El1Attributes::empty());
                    assert_eq!(descriptor.output_address(), PhysicalAddress(0));
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn modify_unmap_compact() {
        let mut idmap = IdMap::new(IdTranslation::new(), 1, El1And0::with_asid(1));
        assert_eq!(idmap.size(), PAGE_SIZE * 512 * 512 * 512);
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules.
        let ttbr = unsafe { idmap.activate() };

        // Map two pages, which will cause subtables to be split out.
        idmap
            .map_fixed_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        // Use `modify_range` to unmap the pages.
        idmap
            .modify_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                &|_, descriptor| descriptor.set(PhysicalAddress(0), El1Attributes::empty()),
            )
            .unwrap();
        // Compact to remove the subtables.
        idmap.compact_subtables();
        // All entries in the top-level table should be 0.
        idmap
            .walk_range(
                &MemoryRegion::new(0, idmap.size()),
                &mut |region, descriptor, level| {
                    assert_eq!(region.len(), PAGE_SIZE * 512 * 512);
                    assert_eq!(descriptor.bits(), 0);
                    assert_eq!(level, 1);
                    Ok(())
                },
            )
            .unwrap();

        unsafe {
            idmap.deactivate(ttbr);
        }
    }

    #[test]
    fn table_sizes() {
        assert_eq!(
            IdMap::<El1And0>::new(IdTranslation::new(), 0, El1And0::with_asid(1)).size(),
            1 << 48
        );
        assert_eq!(
            IdMap::<El1And0>::new(IdTranslation::new(), 1, El1And0::with_asid(1)).size(),
            1 << 39
        );
        assert_eq!(
            IdMap::<El1And0>::new(IdTranslation::new(), 2, El1And0::with_asid(1)).size(),
            1 << 30
        );
        assert_eq!(
            IdMap::<El1And0>::new(IdTranslation::new(), 3, El1And0::with_asid(1)).size(),
            1 << 21
        );
    }

    #[test]
    fn dont_use_l0_block_mapping() {
        // We don't currently support FEAT_LPA2; test that the mappings do not attempt to use huge tables
        let mut idmap = IdMap::new(IdTranslation::new(), 0, El1And0::with_asid(1));
        let block_size = PAGE_SIZE * 512 * 512 * 512; // 512 GiB, corresponding to Level 0
        let range = MemoryRegion::new(0, block_size);

        // The range should map at Level 1 even though it's big enough for Level 0
        idmap
            .map_fixed_range(
                &range,
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        assert_eq!(idmap.root.mapping_level(range.start()), Some(1));

        // The subtable should be cleaned up correctly after unmapping
        idmap
            .map_fixed_range(&range, El1Attributes::empty(), Constraints::empty())
            .unwrap();
        idmap.compact_subtables();
        assert_eq!(idmap.root.mapping_level(range.start()), None);
    }
}
