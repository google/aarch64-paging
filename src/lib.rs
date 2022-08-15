// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! A library to manipulate AArch64 VMSA page tables.
//!
//! Currently it only supports:
//!   - stage 1 page tables
//!   - EL1
//!   - 4 KiB pages
//!
//! Full support is provided for identity mapping ([`IdMap`](idmap::IdMap)) and linear mapping
//! ([`LinearMap`](linearmap::LinearMap)). If you want to use a different mapping scheme, you must
//! provide an implementation of the [`Translation`](paging::Translation) trait and then use
//! [`Mapping`] directly.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "alloc")] {
//! use aarch64_paging::{
//!     idmap::IdMap,
//!     paging::{Attributes, MemoryRegion},
//! };
//!
//! const ASID: usize = 1;
//! const ROOT_LEVEL: usize = 1;
//!
//! // Create a new page table with identity mapping.
//! let mut idmap = IdMap::new(ASID, ROOT_LEVEL);
//! // Map a 2 MiB region of memory as read-only.
//! idmap.map_range(
//!     &MemoryRegion::new(0x80200000, 0x80400000),
//!     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::READ_ONLY,
//! ).unwrap();
//! // Set `TTBR0_EL1` to activate the page table.
//! # #[cfg(target_arch = "aarch64")]
//! idmap.activate();
//! # }
//! ```

#![no_std]

#[cfg(feature = "alloc")]
pub mod idmap;
#[cfg(feature = "alloc")]
pub mod linearmap;
pub mod paging;

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(target_arch = "aarch64")]
use core::arch::asm;
use core::fmt::{self, Display, Formatter};
use paging::{
    Attributes, MemoryRegion, PhysicalAddress, RootTable, Translation, VaRange, VirtualAddress,
};

/// An error attempting to map some range in the page table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MapError {
    /// The address requested to be mapped was out of the range supported by the page table
    /// configuration.
    AddressRange(VirtualAddress),
    /// The address requested to be mapped was not valid for the mapping in use.
    InvalidVirtualAddress(VirtualAddress),
    /// The end of the memory region is before the start.
    RegionBackwards(MemoryRegion),
}

impl Display for MapError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::AddressRange(va) => write!(f, "Virtual address {} out of range", va),
            Self::InvalidVirtualAddress(va) => {
                write!(f, "Invalid virtual address {} for mapping", va)
            }
            Self::RegionBackwards(region) => {
                write!(f, "End of memory region {} is before start.", region)
            }
        }
    }
}

/// Manages a level 1 page table and associated state.
///
/// Mappings should be added with [`map_range`](Self::map_range) before calling
/// [`activate`](Self::activate) to start using the new page table. To make changes which may
/// require break-before-make semantics you must first call [`deactivate`](Self::deactivate) to
/// switch back to a previous static page table, and then `activate` again after making the desired
/// changes.
#[derive(Debug)]
pub struct Mapping<T: Translation + Clone> {
    root: RootTable<T>,
    #[allow(unused)]
    asid: usize,
    #[allow(unused)]
    previous_ttbr: Option<usize>,
}

impl<T: Translation + Clone> Mapping<T> {
    /// Creates a new page table with the given ASID, root level and translation mapping.
    pub fn new(translation: T, asid: usize, rootlevel: usize, va_range: VaRange) -> Self {
        Self {
            root: RootTable::new(translation, rootlevel, va_range),
            asid,
            previous_ttbr: None,
        }
    }

    /// Activates the page table by setting `TTBRn_EL1` to point to it, and saves the previous value
    /// of `TTBRn_EL1` so that it may later be restored by [`deactivate`](Self::deactivate).
    ///
    /// Panics if a previous value of `TTBRn_EL1` is already saved and not yet used by a call to
    /// `deactivate`.
    #[cfg(target_arch = "aarch64")]
    pub fn activate(&mut self) {
        assert!(self.previous_ttbr.is_none());

        let mut previous_ttbr;
        unsafe {
            // Safe because we trust that self.root.to_physical() returns a valid physical address
            // of a page table, and the `Drop` implementation will reset `TTBRn_EL1` before it
            // becomes invalid.
            match self.root.va_range() {
                VaRange::Lower => asm!(
                    "mrs   {previous_ttbr}, ttbr0_el1",
                    "msr   ttbr0_el1, {ttbrval}",
                    "isb",
                    ttbrval = in(reg) self.root.to_physical().0 | (self.asid << 48),
                    previous_ttbr = out(reg) previous_ttbr,
                    options(preserves_flags),
                ),
                VaRange::Upper => asm!(
                    "mrs   {previous_ttbr}, ttbr1_el1",
                    "msr   ttbr1_el1, {ttbrval}",
                    "isb",
                    ttbrval = in(reg) self.root.to_physical().0 | (self.asid << 48),
                    previous_ttbr = out(reg) previous_ttbr,
                    options(preserves_flags),
                ),
            }
        }
        self.previous_ttbr = Some(previous_ttbr);
    }

    /// Deactivates the page table, by setting `TTBRn_EL1` back to the value it had before
    /// [`activate`](Self::activate) was called, and invalidating the TLB for this page table's
    /// configured ASID.
    ///
    /// Panics if there is no saved `TTBRn_EL1` value because `activate` has not previously been
    /// called.
    #[cfg(target_arch = "aarch64")]
    pub fn deactivate(&mut self) {
        unsafe {
            // Safe because this just restores the previously saved value of `TTBRn_EL1`, which must
            // have been valid.
            match self.root.va_range() {
                VaRange::Lower => asm!(
                    "msr   ttbr0_el1, {ttbrval}",
                    "isb",
                    "tlbi  aside1, {asid}",
                    "dsb   nsh",
                    "isb",
                    asid = in(reg) self.asid << 48,
                    ttbrval = in(reg) self.previous_ttbr.unwrap(),
                    options(preserves_flags),
                ),
                VaRange::Upper => asm!(
                    "msr   ttbr1_el1, {ttbrval}",
                    "isb",
                    "tlbi  aside1, {asid}",
                    "dsb   nsh",
                    "isb",
                    asid = in(reg) self.asid << 48,
                    ttbrval = in(reg) self.previous_ttbr.unwrap(),
                    options(preserves_flags),
                ),
            }
        }
        self.previous_ttbr = None;
    }

    /// Maps the given range of virtual addresses to the corresponding range of physical addresses
    /// starting at `pa`, with the given flags.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active.
    pub fn map_range(
        &mut self,
        range: &MemoryRegion,
        pa: PhysicalAddress,
        flags: Attributes,
    ) -> Result<(), MapError> {
        self.root.map_range(range, pa, flags)?;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // Safe because this is just a memory barrier.
            asm!("dsb ishst");
        }
        Ok(())
    }
}

impl<T: Translation + Clone> Drop for Mapping<T> {
    fn drop(&mut self) {
        if self.previous_ttbr.is_some() {
            #[cfg(target_arch = "aarch64")]
            self.deactivate();
        }
    }
}
