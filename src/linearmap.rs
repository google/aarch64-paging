// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with linear mapping.

use crate::{
    paging::{Attributes, MemoryRegion, PhysicalAddress, RootTable, Translation, VirtualAddress},
    AddressRangeError,
};
#[cfg(target_arch = "aarch64")]
use core::arch::asm;

/// Manages a level 1 page-table using linear mapping, where every virtual address is either
/// unmapped or mapped to an IPA with a fixed offset.
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
///     linearmap::LinearMap,
///     paging::{Attributes, MemoryRegion},
/// };
///
/// const ASID: usize = 1;
/// const ROOT_LEVEL: usize = 1;
/// const OFFSET: isize = 4096;
///
/// // Create a new page table with linear mapping.
/// let mut pagetable = LinearMap::new(ASID, ROOT_LEVEL, OFFSET);
/// // Map a 2 MiB region of memory as read-write.
/// pagetable.map_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::EXECUTE_NEVER,
/// ).unwrap();
/// // Set `TTBR0_EL1` to activate the page table.
/// # #[cfg(target_arch = "aarch64")]
/// pagetable.activate();
///
/// // Write something to the memory...
///
/// // Restore `TTBR0_EL1` to its earlier value while we modify the page table.
/// # #[cfg(target_arch = "aarch64")]
/// pagetable.deactivate();
/// // Now change the mapping to read-only and executable.
/// pagetable.map_range(
///     &MemoryRegion::new(0x80200000, 0x80400000),
///     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::READ_ONLY,
/// ).unwrap();
/// # #[cfg(target_arch = "aarch64")]
/// pagetable.activate();
/// ```
#[derive(Debug)]
pub struct LinearMap {
    root: RootTable<LinearTranslation>,
    #[allow(unused)]
    asid: usize,
    #[allow(unused)]
    previous_ttbr: Option<usize>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct LinearTranslation {
    /// The offset from a virtual address to the corresponding (intermediate) physical address.
    offset: isize,
}

impl Translation for LinearTranslation {
    fn virtual_to_physical(&self, va: VirtualAddress) -> PhysicalAddress {
        PhysicalAddress(va.0.wrapping_add(self.offset as usize))
    }

    fn physical_to_virtual(&self, pa: PhysicalAddress) -> VirtualAddress {
        VirtualAddress(pa.0.wrapping_sub(self.offset as usize))
    }
}

impl LinearMap {
    /// Creates a new identity-mapping page table with the given ASID and root level.
    pub fn new(asid: usize, rootlevel: usize, offset: isize) -> LinearMap {
        LinearMap {
            root: RootTable::new(LinearTranslation { offset }, rootlevel),
            asid,
            previous_ttbr: None,
        }
    }

    /// Activates the page table by setting `TTBR0_EL1` to point to it, and saves the previous value
    /// of `TTBR0_EL1` so that it may later be restored by [`deactivate`](Self::deactivate).
    ///
    /// Panics if a previous value of `TTBR0_EL1` is already saved and not yet used by a call to
    /// `deactivate`.
    #[cfg(target_arch = "aarch64")]
    pub fn activate(&mut self) {
        assert!(self.previous_ttbr.is_none());

        let mut previous_ttbr;
        unsafe {
            // Safe because we trust that self.root.to_physical() returns a valid physical address
            // of a page table, and the `Drop` implementation will reset `TTRB0_EL1` before it
            // becomes invalid.
            asm!(
                "mrs   {previous_ttbr}, ttbr0_el1",
                "msr   ttbr0_el1, {ttbrval}",
                "isb",
                ttbrval = in(reg) self.root.to_physical().0 | (self.asid << 48),
                previous_ttbr = out(reg) previous_ttbr,
                options(preserves_flags),
            );
        }
        self.previous_ttbr = Some(previous_ttbr);
    }

    /// Deactivates the page table, by setting `TTBR0_EL1` back to the value it had before
    /// [`activate`](Self::activate) was called, and invalidating the TLB for this page table's
    /// configured ASID.
    ///
    /// Panics if there is no saved `TTRB0_EL1` value because `activate` has not previously been
    /// called.
    #[cfg(target_arch = "aarch64")]
    pub fn deactivate(&mut self) {
        unsafe {
            // Safe because this just restores the previously saved value of `TTBR0_EL1`, which must
            // have been valid.
            asm!(
                "msr   ttbr0_el1, {ttbrval}",
                "isb",
                "tlbi  aside1, {asid}",
                "dsb   nsh",
                "isb",
                asid = in(reg) self.asid << 48,
                ttbrval = in(reg) self.previous_ttbr.unwrap(),
                options(preserves_flags),
            );
        }
        self.previous_ttbr = None;
    }

    /// Maps the given range of virtual addresses to the identical physical addresses with the given
    /// flags.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active.
    pub fn map_range(
        &mut self,
        range: &MemoryRegion,
        flags: Attributes,
    ) -> Result<(), AddressRangeError> {
        self.root.map_range(range, flags)?;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // Safe because this is just a memory barrier.
            asm!("dsb ishst");
        }
        Ok(())
    }
}

impl Drop for LinearMap {
    fn drop(&mut self) {
        if self.previous_ttbr.is_some() {
            #[cfg(target_arch = "aarch64")]
            self.deactivate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paging::PAGE_SIZE;

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
