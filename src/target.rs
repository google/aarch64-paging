// Copyright 2024 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Types for building a static pagetable for some separate target device.
//!
//! See [`TargetAllocator`] for details on how to use it.

use crate::paging::{deallocate, PageTable, PhysicalAddress, Translation};
use alloc::{vec, vec::Vec};
use core::{mem::size_of, ptr::NonNull};
#[cfg(feature = "zerocopy")]
use zerocopy::AsBytes;

/// An implementation of `Translation` which builds a static pagetable to be built into a binary for
/// some target device.
///
/// # Example
///
/// ```
/// use aarch64_paging::{
///     paging::{
///         attributes::Attributes, Constraints, MemoryRegion, PhysicalAddress, RootTable,
///         TranslationRegime, VaRange,
///     },
///     target::TargetAllocator,
/// };
///
/// const ROOT_LEVEL: usize = 1;
///
/// let mut map = RootTable::new(
///     TargetAllocator::new(0x1_0000),
///     ROOT_LEVEL,
///     TranslationRegime::El1And0,
///     VaRange::Lower,
/// );
/// map.map_range(
///     &MemoryRegion::new(0x0, 0x1000),
///     PhysicalAddress(0x4_2000),
///     Attributes::VALID
///         | Attributes::ATTRIBUTE_INDEX_0
///         | Attributes::INNER_SHAREABLE
///         | Attributes::UXN,
///     Constraints::empty(),
/// )
/// .unwrap();
///
/// # #[cfg(feature = "zerocopy")] {
/// let bytes = map.translation().as_bytes();
/// // Build the bytes into a binary image for the target device...
/// # }
/// ```
#[derive(Debug)]
pub struct TargetAllocator {
    base_address: u64,
    allocations: Vec<Option<NonNull<PageTable>>>,
}

impl TargetAllocator {
    /// Creates a new `TargetAllocator` for a page table which will be loaded on the target in a
    /// contiguous block of memory starting at the given address.
    pub fn new(base_address: u64) -> Self {
        Self {
            base_address,
            allocations: vec![],
        }
    }

    fn add_allocation(&mut self, page_table: NonNull<PageTable>) -> usize {
        for (i, allocation) in self.allocations.iter_mut().enumerate() {
            if allocation.is_none() {
                *allocation = Some(page_table);
                return i;
            }
        }
        self.allocations.push(Some(page_table));
        self.allocations.len() - 1
    }

    fn remove_allocation(&mut self, page_table: NonNull<PageTable>) -> bool {
        for allocation in &mut self.allocations {
            if *allocation == Some(page_table) {
                *allocation = None;
                return true;
            }
        }
        false
    }

    /// Returns the full page table as bytes to be loaded into the target device's memory.
    ///
    /// This could be embedded in a binary image for the target.
    #[cfg(feature = "zerocopy")]
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0; self.allocations.len() * size_of::<PageTable>()];
        for (chunk, allocation) in bytes
            .chunks_exact_mut(size_of::<PageTable>())
            .zip(self.allocations.iter())
        {
            if let Some(page_table) = allocation {
                // SAFETY: The pointer is valid because we allocated it in `allocate_table`, and has
                // no aliases for this block.
                let page_table = unsafe { page_table.as_ref() };
                page_table.write_to(chunk).unwrap();
            }
        }
        bytes
    }
}

impl Translation for TargetAllocator {
    fn allocate_table(&mut self) -> (NonNull<PageTable>, PhysicalAddress) {
        let page_table = PageTable::new();
        let index = self.add_allocation(page_table);
        let address = PhysicalAddress(
            usize::try_from(self.base_address).unwrap() + index * size_of::<PageTable>(),
        );
        (page_table, address)
    }

    unsafe fn deallocate_table(&mut self, page_table: NonNull<PageTable>) {
        if !self.remove_allocation(page_table) {
            panic!(
                "dealloc_table called for page table {:?} which isn't in allocations.",
                page_table
            );
        }
        // SAFETY: Our caller promises that the memory was allocated by `allocate_table` on this
        // `TargetAllocator` and not yet deallocated. `allocate_table` used the global allocator
        // and appropriate layout by calling `PageTable::new()`.
        unsafe {
            deallocate(page_table);
        }
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable> {
        self.allocations
            [(pa.0 - usize::try_from(self.base_address).unwrap()) / size_of::<PageTable>()]
        .unwrap()
    }
}

#[cfg(all(test, feature = "zerocopy"))]
mod tests {
    use super::*;
    use crate::paging::{
        attributes::Attributes, Constraints, MemoryRegion, RootTable, TranslationRegime, VaRange,
    };

    const ROOT_LEVEL: usize = 1;

    #[test]
    fn map_one_page() {
        let mut map = RootTable::new(
            TargetAllocator::new(0x1_0000),
            ROOT_LEVEL,
            TranslationRegime::El1And0,
            VaRange::Lower,
        );
        map.map_range(
            &MemoryRegion::new(0x0, 0x1000),
            PhysicalAddress(0x4_2000),
            Attributes::VALID
                | Attributes::ATTRIBUTE_INDEX_0
                | Attributes::INNER_SHAREABLE
                | Attributes::UXN,
            Constraints::empty(),
        )
        .unwrap();

        let bytes = map.translation().as_bytes();
        assert_eq!(bytes.len(), 3 * size_of::<PageTable>());
        // Table mapping for table at 0x01_1000
        assert_eq!(
            bytes[0..8],
            [0x03, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        for byte in &bytes[8..size_of::<PageTable>()] {
            assert_eq!(*byte, 0);
        }
        // Table mapping for table at 0x01_2000
        assert_eq!(
            bytes[size_of::<PageTable>()..size_of::<PageTable>() + 8],
            [0x03, 0x20, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        for byte in &bytes[size_of::<PageTable>() + 8..2 * size_of::<PageTable>()] {
            assert_eq!(*byte, 0);
        }
        // Page mapping for 0x04_2000 with the attributes given above.
        assert_eq!(
            bytes[2 * size_of::<PageTable>()..2 * size_of::<PageTable>() + 8],
            [0x03, 0x23, 0x04, 0x00, 0x00, 0x00, 0x40, 0x00]
        );
        for byte in &bytes[2 * size_of::<PageTable>() + 8..3 * size_of::<PageTable>()] {
            assert_eq!(*byte, 0);
        }
    }
}
