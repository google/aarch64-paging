// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

#[cfg(target_arch = "aarch64")]
use core::arch::asm;
use crate::paging::*;

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
    pub fn new(asid: usize, rootlevel: usize) -> IdMap {
        IdMap {
            root: RootTable::new(rootlevel),
            asid,
        }
    }

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
