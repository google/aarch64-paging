// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with linear mapping.
//!
//! See [`LinearMap`] for details on how to use it.

use crate::{
    MapError, Mapping,
    descriptor::{PagingAttributes, PhysicalAddress, VirtualAddress},
    paging::{
        FixedTranslation, PAGE_SIZE, PageTable, Translation, TranslationRegime, deallocate,
        is_aligned,
    },
};
use core::marker::PhantomData;
use core::ptr::NonNull;

/// Linear mapping, where every virtual address is either unmapped or mapped to an IPA with a fixed
/// offset.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LinearTranslation<A: PagingAttributes> {
    /// The offset from a virtual address to the corresponding (intermediate) physical address.
    offset: isize,
    _phantom: PhantomData<A>,
}

impl<A: PagingAttributes> LinearTranslation<A> {
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
        Self {
            offset,
            _phantom: PhantomData,
        }
    }
}

impl<A: PagingAttributes> Translation<A> for LinearTranslation<A> {
    fn allocate_table(&mut self) -> (NonNull<PageTable<A>>, PhysicalAddress) {
        let table = PageTable::new();
        // Assume that the same linear mapping is used everywhere.
        let va = VirtualAddress(table.as_ptr() as usize);

        let pa = self.virtual_to_physical(va).expect(
            "Allocated subtable with virtual address which doesn't correspond to any physical address."
        );
        (table, pa)
    }

    unsafe fn deallocate_table(&mut self, page_table: NonNull<PageTable<A>>) {
        // SAFETY: Our caller promises that the memory was allocated by `allocate_table` on this
        // `LinearTranslation` and not yet deallocated. `allocate_table` used the global allocator
        // and appropriate layout by calling `PageTable::new()`.
        unsafe {
            deallocate(page_table);
        }
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable<A>> {
        let signed_pa = pa.0 as isize;
        if signed_pa < 0 {
            panic!("Invalid physical address {} for pagetable", pa);
        }
        if let Some(va) = signed_pa.checked_sub(self.offset) {
            if let Some(ptr) = NonNull::new(va as *mut PageTable<A>) {
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

impl<A: PagingAttributes> FixedTranslation for LinearTranslation<A> {
    fn virtual_to_physical(&self, va: VirtualAddress) -> Result<PhysicalAddress, MapError> {
        if let Some(pa) = checked_add_to_unsigned(va.0 as isize, self.offset) {
            Ok(PhysicalAddress(pa))
        } else {
            Err(MapError::InvalidVirtualAddress(va))
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
pub type LinearMap<R> = Mapping<LinearTranslation<<R as TranslationRegime>::Attributes>, R>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlb::{self, EmulatedLinearMap};
    use crate::{
        MapError,
        descriptor::El1Attributes,
        paging::{BITS_PER_LEVEL, Constraints, El1And0, MemoryRegion, PAGE_SIZE, VaRange},
    };

    const MAX_ADDRESS_FOR_ROOT_LEVEL_1: usize = 1 << 39;
    const GIB_512_S: isize = 512 * 1024 * 1024 * 1024;
    const GIB_512: usize = 512 * 1024 * 1024 * 1024;
    const NORMAL_CACHEABLE: El1Attributes =
        El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE);

    #[test]
    fn map_valid() {
        // A single byte at the start of the address space.
        let mut pagetable =
            tlb::mapping_with_asid(LinearTranslation::new(4096), 1, 1, El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(0, 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable =
            tlb::mapping_with_asid(LinearTranslation::new(4096), 1, 1, El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(0, PAGE_SIZE * 2),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable =
            tlb::mapping_with_asid(LinearTranslation::new(4096), 1, 1, El1And0, VaRange::Lower);
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // The entire valid address space. Use an offset that is a multiple of the level 2 block
        // size to avoid mapping everything as pages as that is really slow.
        const LEVEL_2_BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(LEVEL_2_BLOCK_SIZE as isize),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(0, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );
    }

    #[test]
    fn map_valid_negative_offset() {
        // A single byte which maps to IPA 0.
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(-(PAGE_SIZE as isize)),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE + 1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // Two pages at the start of the address space.
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(-(PAGE_SIZE as isize)),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(PAGE_SIZE, PAGE_SIZE * 3),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // A single byte at the end of the address space.
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(-(PAGE_SIZE as isize)),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1 - 1,
                    MAX_ADDRESS_FOR_ROOT_LEVEL_1
                ),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );

        // The entire valid address space. Use an offset that is a multiple of the level 2 block
        // size to avoid mapping everything as pages as that is really slow.
        const LEVEL_2_BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(-(LEVEL_2_BLOCK_SIZE as isize)),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(LEVEL_2_BLOCK_SIZE, MAX_ADDRESS_FOR_ROOT_LEVEL_1),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );
    }

    #[test]
    fn map_top_of_upper_range() {
        // With 4 levels of page tables, the topmost 512 GiB of the address space is the last level 0
        // chunk, which ends at 2^64. Map a region there, as a kernel linked at
        // 0xffff_ffff_8000_0000 would.
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(GIB_512_S),
            1,
            0,
            El1And0,
            VaRange::Upper,
        );
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(0xffff_ffff_8000_0000, 0xffff_ffff_8020_0000),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            ),
            Ok(())
        );
    }

    #[test]
    fn unmap_and_compact_top_of_upper_range() {
        // The topmost tables of the upper VA range cover ranges which end at 2^64. Unmap a whole
        // level 3 table's worth of pages near the top while the page table is live, and then
        // compact away the empty tables above it.
        const BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let region =
            MemoryRegion::new(usize::MAX - 2 * BLOCK_SIZE + 1, usize::MAX - BLOCK_SIZE + 1);
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(GIB_512_S),
            1,
            0,
            El1And0,
            VaRange::Upper,
        );
        pagetable
            .map_fixed_range(
                &region,
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::NO_BLOCK_MAPPINGS,
            )
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules and TLB maintenance.
        let ttbr = unsafe { pagetable.activate() };

        pagetable
            .map_fixed_range(&region, El1Attributes::empty(), Constraints::empty())
            .unwrap();
        pagetable.compact_subtables();

        // SAFETY: This doesn't actually deactivate the page table in tests.
        unsafe { pagetable.deactivate(ttbr) };
        pagetable
            .walk_range(&region, &mut |_, descriptor, level| {
                assert_eq!(level, 0);
                assert_eq!(descriptor.bits(), 0);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn map_out_of_range() {
        let mut pagetable =
            tlb::mapping_with_asid(LinearTranslation::new(4096), 1, 1, El1And0, VaRange::Lower);

        // One byte, just past the edge of the valid range.
        assert_eq!(
            pagetable.map_fixed_range(
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
            pagetable.map_fixed_range(
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
    fn map_invalid_offset() {
        let mut pagetable =
            tlb::mapping_with_asid(LinearTranslation::new(-4096), 1, 1, El1And0, VaRange::Lower);

        // One byte, with an offset which would map it to a negative IPA.
        assert_eq!(
            pagetable.map_fixed_range(
                &MemoryRegion::new(0, 1),
                NORMAL_CACHEABLE,
                Constraints::empty(),
            ),
            Err(MapError::InvalidVirtualAddress(VirtualAddress(0)))
        );
    }

    #[test]
    fn physical_address_in_range_ttbr0() {
        let translation = LinearTranslation::<El1Attributes>::new(4096);
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(8192)),
            NonNull::new(4096 as *mut PageTable<El1Attributes>).unwrap(),
        );
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(GIB_512 + 4096)),
            NonNull::new(GIB_512 as *mut PageTable<El1Attributes>).unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn physical_address_to_zero_ttbr0() {
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(4096);
        translation.physical_to_virtual(PhysicalAddress(4096));
    }

    #[test]
    #[should_panic]
    fn physical_address_out_of_range_ttbr0() {
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(4096);
        translation.physical_to_virtual(PhysicalAddress(-4096_isize as usize));
    }

    #[test]
    fn physical_address_in_range_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to one page above the bottom
        // of physical address space.
        let translation = LinearTranslation::new(GIB_512_S + 4096);
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(8192)),
            NonNull::new((4096 - GIB_512_S) as *mut PageTable<El1Attributes>).unwrap(),
        );
        assert_eq!(
            translation.physical_to_virtual(PhysicalAddress(GIB_512)),
            NonNull::new(-4096_isize as *mut PageTable<El1Attributes>).unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn physical_address_to_zero_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to the bottom of physical
        // address space.
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(GIB_512_S);
        translation.physical_to_virtual(PhysicalAddress(GIB_512));
    }

    #[test]
    #[should_panic]
    fn physical_address_out_of_range_ttbr1() {
        // Map the 512 GiB region at the top of virtual address space to the bottom of physical
        // address space.
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(GIB_512_S);
        translation.physical_to_virtual(PhysicalAddress(-4096_isize as usize));
    }

    #[test]
    fn virtual_address_out_of_range() {
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(-4096);
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
        let translation: LinearTranslation<El1Attributes> = LinearTranslation::new(GIB_512_S);

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
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(1 << 30),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        pagetable
            .map_fixed_range(
                &MemoryRegion::new(0, 1 << 30),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        assert_eq!(pagetable.root.mapping_level(VirtualAddress(0)), Some(1));

        // ...but not when it is not.
        let mut pagetable = tlb::mapping_with_asid(
            LinearTranslation::new(1 << 29),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        pagetable
            .map_fixed_range(
                &MemoryRegion::new(0, 1 << 30),
                NORMAL_CACHEABLE | El1Attributes::VALID | El1Attributes::ACCESSED,
                Constraints::empty(),
            )
            .unwrap();
        assert_eq!(pagetable.root.mapping_level(VirtualAddress(0)), Some(2));
    }

    fn make_map() -> EmulatedLinearMap<El1And0> {
        let mut lmap =
            tlb::mapping_with_asid(LinearTranslation::new(4096), 1, 1, El1And0, VaRange::Lower);
        // Mapping VA range 0x0 - 0x2000 to PA range 0x1000 - 0x3000
        lmap.map_fixed_range(
            &MemoryRegion::new(0, PAGE_SIZE * 2),
            NORMAL_CACHEABLE,
            Constraints::empty(),
        )
        .unwrap();
        lmap
    }

    #[test]
    fn update_backwards_range() {
        let mut lmap = make_map();
        assert!(
            lmap.modify_range(&MemoryRegion::new(PAGE_SIZE * 2, 1), &|_range, entry| {
                entry.modify_flags(
                    El1Attributes::SWFLAG_0,
                    El1Attributes::from_bits(0usize).unwrap(),
                )
            },)
                .is_err()
        );
    }

    #[test]
    fn update_range() {
        let mut lmap = make_map();
        lmap.modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|_range, entry| {
            if !entry.is_table() {
                entry.modify_flags(
                    El1Attributes::SWFLAG_0,
                    El1Attributes::from_bits(0usize).unwrap(),
                )?;
            }
            Ok(())
        })
        .unwrap();
        lmap.modify_range(&MemoryRegion::new(1, PAGE_SIZE), &|range, entry| {
            if !entry.is_table() {
                assert!(entry.flags().contains(El1Attributes::SWFLAG_0));
                assert_eq!(range.end() - range.start(), PAGE_SIZE);
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn breakup_invalid_block() {
        const BLOCK_RANGE: usize = 0x200000;

        let mut lmap = tlb::mapping_with_asid(
            LinearTranslation::new(0x1000),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        lmap.map_fixed_range(
            &MemoryRegion::new(0, BLOCK_RANGE),
            NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::SWFLAG_0,
            Constraints::empty(),
        )
        .unwrap();
        lmap.map_fixed_range(
            &MemoryRegion::new(0, PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
            Constraints::empty(),
        )
        .unwrap();
        lmap.modify_range(&MemoryRegion::new(0, BLOCK_RANGE), &|range, entry| {
            if entry.level() == 3 {
                let has_swflag = entry.flags().contains(El1Attributes::SWFLAG_0);
                let is_first_page = range.start().0 == 0usize;
                assert!(has_swflag != is_first_page);
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[should_panic]
    fn split_live_block_mapping() -> () {
        const BLOCK_SIZE: usize = PAGE_SIZE << BITS_PER_LEVEL;
        let mut lmap = tlb::mapping_with_asid(
            LinearTranslation::new(BLOCK_SIZE as isize),
            1,
            1,
            El1And0,
            VaRange::Lower,
        );
        lmap.map_fixed_range(
            &MemoryRegion::new(0, BLOCK_SIZE),
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
        let ttbr = unsafe { lmap.activate() };
        lmap.map_fixed_range(
            &MemoryRegion::new(0, PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::READ_ONLY
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
            Constraints::empty(),
        )
        .unwrap();
        lmap.map_fixed_range(
            &MemoryRegion::new(PAGE_SIZE, 2 * PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::READ_ONLY
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
            Constraints::empty(),
        )
        .unwrap();
        let r = lmap.map_fixed_range(
            &MemoryRegion::new(PAGE_SIZE, 2 * PAGE_SIZE),
            NORMAL_CACHEABLE
                | El1Attributes::NON_GLOBAL
                | El1Attributes::VALID
                | El1Attributes::ACCESSED,
            Constraints::empty(),
        );
        unsafe { lmap.deactivate(ttbr) };
        r.unwrap();
    }
}
