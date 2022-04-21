// SPDX-License-Identifier: GPL-2.0
// Copyright 2022 Google LLC
// Author: Ard Biesheuvel <ardb@google.com>

use crate::paging::*;
use core::arch::asm;

#[derive(Debug)]
pub struct IdMap {
    root: RootTable,
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

    pub fn activate(&self) {
        unsafe {
            // inline asm is unsafe
            asm!(
                "msr   ttbr0_el1, {ttbrval}",
                "isb",
                ttbrval = in(reg) self.root.to_physical::<Self>().0 | (self.asid << 48),
                options(preserves_flags),
            );
        }
    }

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
        self.root.map_range::<Self>(range, flags);
        unsafe {
            asm!("dsb ishst");
        }
    }
}

impl Drop for IdMap {
    fn drop(&mut self) {
        self.deactivate();
    }
}
