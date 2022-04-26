// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Functionality for managing page tables with identity mapping.

#[cfg(target_arch = "aarch64")]
use core::arch::asm;
use crate::paging::*;

/// Manages a level 1 page-table using identity mapping, where every virtual address is either
/// unmapped or mapped to the identical IPA.
#[derive(Debug)]
pub struct IdMap {
    root: RootTable<IdMap>,
    #[allow(unused)]
    asid: usize,
}

impl Translation for IdMap {
    fn virtual_to_physical(va: VirtualAddress) -> PhysicalAddress {
        PhysicalAddress(va.0)
    }

    fn physical_to_virtual(pa: PhysicalAddress) -> VirtualAddress {
        VirtualAddress(pa.0)
    }
}

impl IdMap {
    /// Creates a new identity-mapping page table with the given ASID and root level.
    pub fn new(asid: usize, rootlevel: usize) -> IdMap {
        IdMap {
            root: RootTable::new(rootlevel),
            asid,
        }
    }

    /// Activates the page table by setting `TTBR_EL1` to point to it.
    #[cfg(target_arch = "aarch64")]
    pub fn activate(&self) {
        unsafe {
            // inline asm is unsafe
            asm!(
                "msr   ttbr0_el1, {ttbrval}",
                "isb",
                ttbrval = in(reg) self.root.to_physical().0 | (self.asid << 48),
                options(preserves_flags),
            );
        }
    }

    /// Deactivates the page table, by setting `TTBR_EL1` to point to a statically configured
    /// `idmap` instead, and invalidating the TLB for this page table's configured ASID.
    #[cfg(target_arch = "aarch64")]
    pub fn deactivate(&self) {
        unsafe {
            // inline asm is unsafe
            asm!(
                "adrp  {ttbrval}, idmap",
                "msr   ttbr0_el1, {ttbrval}",
                "isb",
                "tlbi  aside1, {asid}",
                "dsb   nsh",
                "isb",
                asid = in(reg) self.asid << 48,
                ttbrval = lateout(reg) _,
                options(preserves_flags),
            );
        }
    }

    /// Maps the given range of virtual addresses to the identical physical addresses with the given
    /// flags.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active.
    pub fn map_range(&mut self, range: &MemoryRegion, flags: Attributes) {
        self.root.map_range(range, flags);
        #[cfg(target_arch = "aarch64")]
        unsafe {
            asm!("dsb ishst");
        }
    }
}

impl Drop for IdMap {
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        self.deactivate();
    }
}
