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
//! ```no_run
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
//!     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::READ_ONLY | Attributes::VALID,
//! ).unwrap();
//! // Set `TTBR0_EL1` to activate the page table.
//! idmap.activate();
//! # }
//! ```

#![no_std]
#![deny(clippy::undocumented_unsafe_blocks)]

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
    Attributes, Constraints, Descriptor, MemoryRegion, PhysicalAddress, RootTable, Translation,
    VaRange, VirtualAddress,
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
    /// There was an error while updating a page table entry.
    PteUpdateFault(Descriptor),
    /// The requested flags are not supported for this mapping
    InvalidFlags(Attributes),
    /// Updating the range violates break-before-make rules and the mapping is live
    BreakBeforeMakeViolation(MemoryRegion),
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
            Self::PteUpdateFault(desc) => {
                write!(f, "Error updating page table entry {:?}", desc)
            }
            Self::InvalidFlags(flags) => {
                write!(f, "Flags {flags:?} unsupported for mapping.")
            }
            Self::BreakBeforeMakeViolation(region) => {
                write!(f, "Cannot remap region {region} while translation is live.")
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

    /// Returns whether this mapping is currently active.
    pub fn active(&self) -> bool {
        self.previous_ttbr.is_some()
    }

    /// Activates the page table by setting `TTBRn_EL1` to point to it, and saves the previous value
    /// of `TTBRn_EL1` so that it may later be restored by [`deactivate`](Self::deactivate).
    ///
    /// Panics if a previous value of `TTBRn_EL1` is already saved and not yet used by a call to
    /// `deactivate`.
    ///
    /// In test builds or builds that do not target aarch64, the `TTBRn_EL1` access is omitted.
    pub fn activate(&mut self) {
        assert!(!self.active());

        #[allow(unused)]
        let mut previous_ttbr = usize::MAX;

        #[cfg(all(not(test), target_arch = "aarch64"))]
        // SAFETY: Safe because we trust that self.root.to_physical() returns a valid physical
        // address of a page table, and the `Drop` implementation will reset `TTBRn_EL1` before it
        // becomes invalid.
        unsafe {
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
    ///
    /// In test builds or builds that do not target aarch64, the `TTBRn_EL1` access is omitted.
    pub fn deactivate(&mut self) {
        assert!(self.active());

        #[cfg(all(not(test), target_arch = "aarch64"))]
        // SAFETY: Safe because this just restores the previously saved value of `TTBRn_EL1`, which
        // must have been valid.
        unsafe {
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

    /// Checks whether the given range can be mapped or updated while the translation is live,
    /// without violating architectural break-before-make (BBM) requirements.
    fn check_range_bbm<F>(&self, range: &MemoryRegion, updater: &F) -> Result<(), MapError>
    where
        F: Fn(&MemoryRegion, &mut Descriptor, usize) -> Result<(), ()> + ?Sized,
    {
        self.walk_range(
            range,
            &mut |mr: &MemoryRegion, d: &Descriptor, level: usize| {
                if d.is_valid() {
                    if !mr.is_block(level) {
                        // Cannot split a live block mapping
                        return Err(());
                    }

                    // Get the new flags and output address for this descriptor by applying
                    // the updater function to a copy
                    let (flags, oa) = {
                        let mut dd = *d;
                        updater(mr, &mut dd, level)?;
                        (dd.flags().ok_or(())?, dd.output_address())
                    };

                    if !flags.contains(Attributes::VALID) {
                        // Removing the valid bit is always ok
                        return Ok(());
                    }

                    if oa != d.output_address() {
                        // Cannot change output address on a live mapping
                        return Err(());
                    }

                    let desc_flags = d.flags().unwrap();

                    if (desc_flags ^ flags).intersects(Attributes::NORMAL) {
                        // Cannot change memory type
                        return Err(());
                    }

                    if (desc_flags - flags).contains(Attributes::NON_GLOBAL) {
                        // Cannot convert from non-global to global
                        return Err(());
                    }
                }
                Ok(())
            },
        )
        .map_err(|_| MapError::BreakBeforeMakeViolation(range.clone()))?;
        Ok(())
    }

    /// Maps the given range of virtual addresses to the corresponding range of physical addresses
    /// starting at `pa`, with the given flags, taking the given constraints into account.
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
    pub fn map_range(
        &mut self,
        range: &MemoryRegion,
        pa: PhysicalAddress,
        flags: Attributes,
        constraints: Constraints,
    ) -> Result<(), MapError> {
        if self.active() {
            let c = |mr: &MemoryRegion, d: &mut Descriptor, lvl: usize| {
                let mask = !(paging::granularity_at_level(lvl) - 1);
                let pa = (mr.start() - range.start() + pa.0) & mask;
                d.set(PhysicalAddress(pa), flags);
                Ok(())
            };
            self.check_range_bbm(range, &c)?;
        }
        self.root.map_range(range, pa, flags, constraints)?;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: Safe because this is just a memory barrier.
        unsafe {
            asm!("dsb ishst");
        }
        Ok(())
    }

    /// Applies the provided updater function to a number of PTEs corresponding to a given memory range.
    ///
    /// This may involve splitting block entries if the provided range is not currently mapped
    /// down to its precise boundaries. For visiting all the descriptors covering a memory range
    /// without potential splitting (and no descriptor updates), use
    /// [`walk_range`](Self::walk_range) instead.
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
        if self.active() {
            self.check_range_bbm(range, f)?;
        }
        self.root.modify_range(range, f)?;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: Safe because this is just a memory barrier.
        unsafe {
            asm!("dsb ishst");
        }
        Ok(())
    }

    /// Applies the provided function to a number of PTEs corresponding to a given memory range.
    ///
    /// The virtual address range passed to the callback function may be expanded compared to the
    /// `range` parameter, due to alignment to block boundaries.
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
        self.root.walk_range(range, f)
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
