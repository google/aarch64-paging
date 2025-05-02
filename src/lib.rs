// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! A library to manipulate AArch64 VMSA page tables.
//!
//! Currently it only supports:
//!   - stage 1 page tables
//!   - 4 KiB pages
//!   - EL3, NS-EL2, NS-EL2&0 and NS-EL1&0 translation regimes
//!
//! Full support is provided for identity mapping ([`IdMap`](idmap::IdMap)) and linear mapping
//! ([`LinearMap`](linearmap::LinearMap)). If you want to use a different mapping scheme, you must
//! provide an implementation of the [`Translation`] trait and then use [`Mapping`] directly.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "alloc")] {
//! use aarch64_paging::{
//!     idmap::IdMap,
//!     paging::{Attributes, MemoryRegion, TranslationRegime},
//! };
//!
//! const ASID: usize = 1;
//! const ROOT_LEVEL: usize = 1;
//! const NORMAL_CACHEABLE: Attributes = Attributes::ATTRIBUTE_INDEX_1.union(Attributes::INNER_SHAREABLE);
//!
//! // Create a new EL1 page table with identity mapping.
//! let mut idmap = IdMap::new(ASID, ROOT_LEVEL, TranslationRegime::El1And0);
//! // Map a 2 MiB region of memory as read-write.
//! idmap.map_range(
//!     &MemoryRegion::new(0x80200000, 0x80400000),
//!     NORMAL_CACHEABLE | Attributes::NON_GLOBAL | Attributes::VALID | Attributes::ACCESSED,
//! ).unwrap();
//! // SAFETY: Everything the program uses is within the 2 MiB region mapped above.
//! unsafe {
//!     // Set `TTBR0_EL1` to activate the page table.
//!     idmap.activate();
//! }
//! # }
//! ```

#![no_std]
#![deny(clippy::undocumented_unsafe_blocks)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "alloc")]
pub mod idmap;
#[cfg(feature = "alloc")]
pub mod linearmap;
pub mod mair;
pub mod paging;
#[cfg(feature = "alloc")]
pub mod target;

#[cfg(any(test, feature = "alloc"))]
extern crate alloc;

#[cfg(target_arch = "aarch64")]
use core::arch::asm;
use paging::{
    Attributes, Constraints, Descriptor, MemoryRegion, PhysicalAddress, RootTable, Translation,
    TranslationRegime, VaRange, VirtualAddress,
};
use thiserror::Error;

/// An error attempting to map some range in the page table.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum MapError {
    /// The address requested to be mapped was out of the range supported by the page table
    /// configuration.
    #[error("Virtual address {0} out of range")]
    AddressRange(VirtualAddress),
    /// The address requested to be mapped was not valid for the mapping in use.
    #[error("Invalid virtual address {0} for mapping")]
    InvalidVirtualAddress(VirtualAddress),
    /// The end of the memory region is before the start.
    #[error("End of memory region {0} is before start.")]
    RegionBackwards(MemoryRegion),
    /// There was an error while updating a page table entry.
    #[error("Error updating page table entry {0:?}")]
    PteUpdateFault(Descriptor),
    /// The requested flags are not supported for this mapping
    #[error("Flags {0:?} unsupported for mapping.")]
    InvalidFlags(Attributes),
    /// Updating the range violates break-before-make rules and the mapping is live
    #[error("Cannot remap region {0} while translation is live.")]
    BreakBeforeMakeViolation(MemoryRegion),
}

/// Manages a level 1 page table and associated state.
///
/// Mappings should be added with [`map_range`](Self::map_range) before calling
/// [`activate`](Self::activate) to start using the new page table. To make changes which may
/// require break-before-make semantics you must first call [`deactivate`](Self::deactivate) to
/// switch back to a previous static page table, and then `activate` again after making the desired
/// changes.
#[derive(Debug)]
pub struct Mapping<T: Translation> {
    root: RootTable<T>,
    asid: usize,
    #[allow(unused)]
    previous_ttbr: Option<usize>,
}

impl<T: Translation> Mapping<T> {
    /// Creates a new page table with the given ASID, root level and translation mapping.
    pub fn new(
        translation: T,
        asid: usize,
        rootlevel: usize,
        translation_regime: TranslationRegime,
        va_range: VaRange,
    ) -> Self {
        if !translation_regime.supports_asid() && asid != 0 {
            panic!("{:?} doesn't support ASID, must be 0.", translation_regime);
        }
        Self {
            root: RootTable::new(translation, rootlevel, translation_regime, va_range),
            asid,
            previous_ttbr: None,
        }
    }

    /// Returns a reference to the translation used for this page table.
    pub fn translation(&self) -> &T {
        self.root.translation()
    }

    /// Returns whether this mapping is currently active.
    pub fn active(&self) -> bool {
        self.previous_ttbr.is_some()
    }

    /// Returns the size in bytes of the virtual address space which can be mapped in this page
    /// table.
    ///
    /// This is a function of the chosen root level.
    pub fn size(&self) -> usize {
        self.root.size()
    }

    /// Activates the page table by setting `TTBRn_ELx` to point to it, and saves the previous value
    /// of `TTBRn_ELx` so that it may later be restored by [`deactivate`](Self::deactivate).
    ///
    /// Panics if a previous value of `TTBRn_ELx` is already saved and not yet used by a call to
    /// `deactivate`.
    ///
    /// In test builds or builds that do not target aarch64, the `TTBRn_ELx` access is omitted.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the page table doesn't unmap any memory which the program is
    /// using, or introduce aliases which break Rust's aliasing rules. The page table must not be
    /// dropped as long as its mappings are required, as it will automatically be deactivated when
    /// it is dropped.
    pub unsafe fn activate(&mut self) {
        assert!(!self.active());

        #[allow(unused)]
        let mut previous_ttbr = usize::MAX;

        #[cfg(all(not(test), target_arch = "aarch64"))]
        // SAFETY: We trust that self.root_address() returns a valid physical address of a page
        // table, and the `Drop` implementation will reset `TTBRn_ELx` before it becomes invalid.
        unsafe {
            match (self.root.translation_regime(), self.root.va_range()) {
                (TranslationRegime::El1And0, VaRange::Lower) => asm!(
                    "mrs   {previous_ttbr}, ttbr0_el1",
                    "msr   ttbr0_el1, {ttbrval}",
                    "isb",
                    ttbrval = in(reg) self.root_address().0 | (self.asid << 48),
                    previous_ttbr = out(reg) previous_ttbr,
                    options(preserves_flags),
                ),
                (TranslationRegime::El1And0, VaRange::Upper) => asm!(
                    "mrs   {previous_ttbr}, ttbr1_el1",
                    "msr   ttbr1_el1, {ttbrval}",
                    "isb",
                    ttbrval = in(reg) self.root_address().0 | (self.asid << 48),
                    previous_ttbr = out(reg) previous_ttbr,
                    options(preserves_flags),
                ),
                (TranslationRegime::El2And0, VaRange::Lower) => asm!(
                    "mrs   {previous_ttbr}, ttbr0_el2",
                    "msr   ttbr0_el2, {ttbrval}",
                    "isb",
                    ttbrval = in(reg) self.root_address().0 | (self.asid << 48),
                    previous_ttbr = out(reg) previous_ttbr,
                    options(preserves_flags),
                ),
                (TranslationRegime::El2And0, VaRange::Upper) => asm!(
                    "mrs   {previous_ttbr}, s3_4_c2_c0_1", // ttbr1_el2
                    "msr   s3_4_c2_c0_1, {ttbrval}",
                    "isb",
                    ttbrval = in(reg) self.root_address().0 | (self.asid << 48),
                    previous_ttbr = out(reg) previous_ttbr,
                    options(preserves_flags),
                ),
                (TranslationRegime::El2, VaRange::Lower) => asm!(
                    "mrs   {previous_ttbr}, ttbr0_el2",
                    "msr   ttbr0_el2, {ttbrval}",
                    "isb",
                    ttbrval = in(reg) self.root_address().0,
                    previous_ttbr = out(reg) previous_ttbr,
                    options(preserves_flags),
                ),
                (TranslationRegime::El3, VaRange::Lower) => asm!(
                    "mrs   {previous_ttbr}, ttbr0_el3",
                    "msr   ttbr0_el3, {ttbrval}",
                    "isb",
                    ttbrval = in(reg) self.root_address().0,
                    previous_ttbr = out(reg) previous_ttbr,
                    options(preserves_flags),
                ),
                _ => {
                    panic!("Invalid combination of exception level and VA range.");
                }
            }
        }
        self.mark_active(previous_ttbr);
    }

    /// Deactivates the page table, by setting `TTBRn_ELx` back to the value it had before
    /// [`activate`](Self::activate) was called, and invalidating the TLB for this page table's
    /// configured ASID.
    ///
    /// Panics if there is no saved `TTBRn_ELx` value because `activate` has not previously been
    /// called.
    ///
    /// In test builds or builds that do not target aarch64, the `TTBRn_ELx` access is omitted.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the previous page table which this is switching back to doesn't
    /// unmap any memory which the program is using.
    pub unsafe fn deactivate(&mut self) {
        assert!(self.active());

        #[cfg(all(not(test), target_arch = "aarch64"))]
        // SAFETY: This just restores the previously saved value of `TTBRn_ELx`, which must have
        // been valid.
        unsafe {
            match (self.root.translation_regime(), self.root.va_range()) {
                (TranslationRegime::El1And0, VaRange::Lower) => asm!(
                    "msr   ttbr0_el1, {ttbrval}",
                    "isb",
                    "tlbi  aside1, {asid}",
                    "dsb   nsh",
                    "isb",
                    asid = in(reg) self.asid << 48,
                    ttbrval = in(reg) self.previous_ttbr.unwrap(),
                    options(preserves_flags),
                ),
                (TranslationRegime::El1And0, VaRange::Upper) => asm!(
                    "msr   ttbr1_el1, {ttbrval}",
                    "isb",
                    "tlbi  aside1, {asid}",
                    "dsb   nsh",
                    "isb",
                    asid = in(reg) self.asid << 48,
                    ttbrval = in(reg) self.previous_ttbr.unwrap(),
                    options(preserves_flags),
                ),
                (TranslationRegime::El2And0, VaRange::Lower) => asm!(
                    "msr   ttbr0_el2, {ttbrval}",
                    "isb",
                    "tlbi  aside1, {asid}",
                    "dsb   nsh",
                    "isb",
                    asid = in(reg) self.asid << 48,
                    ttbrval = in(reg) self.previous_ttbr.unwrap(),
                    options(preserves_flags),
                ),
                (TranslationRegime::El2And0, VaRange::Upper) => asm!(
                    "msr   s3_4_c2_c0_1, {ttbrval}", // ttbr1_el2
                    "isb",
                    "tlbi  aside1, {asid}",
                    "dsb   nsh",
                    "isb",
                    asid = in(reg) self.asid << 48,
                    ttbrval = in(reg) self.previous_ttbr.unwrap(),
                    options(preserves_flags),
                ),
                (TranslationRegime::El2, VaRange::Lower) => {
                    panic!("EL2 page table can't safety be deactivated.");
                }
                (TranslationRegime::El3, VaRange::Lower) => {
                    panic!("EL3 page table can't safety be deactivated.");
                }
                _ => {
                    panic!("Invalid combination of exception level and VA range.");
                }
            }
        }
        self.mark_inactive();
    }

    /// Checks whether the given range can be mapped or updated while the translation is live,
    /// without violating architectural break-before-make (BBM) requirements.
    fn check_range_bbm<F>(&self, range: &MemoryRegion, updater: &F) -> Result<(), MapError>
    where
        F: Fn(&MemoryRegion, &mut Descriptor, usize) -> Result<(), ()> + ?Sized,
    {
        self.root.visit_range(
            range,
            &mut |mr: &MemoryRegion, d: &Descriptor, level: usize| {
                if d.is_valid() {
                    let err = MapError::BreakBeforeMakeViolation(mr.clone());

                    if !mr.is_block(level) {
                        // Cannot split a live block mapping
                        return Err(err);
                    }

                    // Get the new flags and output address for this descriptor by applying
                    // the updater function to a copy
                    let (flags, oa) = {
                        let mut dd = *d;
                        updater(mr, &mut dd, level).or(Err(err.clone()))?;
                        (dd.flags(), dd.output_address())
                    };

                    if !flags.contains(Attributes::VALID) {
                        // Removing the valid bit is always ok
                        return Ok(());
                    }

                    if oa != d.output_address() {
                        // Cannot change output address on a live mapping
                        return Err(err);
                    }

                    let desc_flags = d.flags();

                    if (desc_flags ^ flags).intersects(
                        Attributes::ATTRIBUTE_INDEX_MASK | Attributes::SHAREABILITY_MASK,
                    ) {
                        // Cannot change memory type
                        return Err(err);
                    }

                    if (desc_flags - flags).contains(Attributes::NON_GLOBAL) {
                        // Cannot convert from non-global to global
                        return Err(err);
                    }
                }
                Ok(())
            },
        )
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
        // SAFETY: This is just a memory barrier.
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
        // SAFETY: This is just a memory barrier.
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

    /// Returns the physical address of the root table.
    ///
    /// This may be used to activate the page table by setting the appropriate TTBRn_ELx if you wish
    /// to do so yourself rather than by calling [`activate`](Self::activate). Make sure to call
    /// [`mark_active`](Self::mark_active) after doing so.
    pub fn root_address(&self) -> PhysicalAddress {
        self.root.to_physical()
    }

    /// Returns the ASID of the page table.
    pub fn asid(&self) -> usize {
        self.asid
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
        self.previous_ttbr = Some(previous_ttbr);
    }

    /// Marks the page table as inactive.
    ///
    /// This may be called after manually disabling the use of the page table, such as by setting
    /// the relevant TTBR to a different address.
    ///
    /// It is called automatically by [`deactivate`](Self::deactivate).
    pub fn mark_inactive(&mut self) {
        self.previous_ttbr = None;
    }
}

impl<T: Translation> Drop for Mapping<T> {
    fn drop(&mut self) {
        if self.previous_ttbr.is_some() {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: When activate was called the caller promised that they wouldn't drop the page
            // table until its mappings were no longer needed.
            unsafe {
                self.deactivate();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "alloc")]
    use self::idmap::IdTranslation;
    #[cfg(feature = "alloc")]
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    #[should_panic]
    fn no_el2_asid() {
        Mapping::new(IdTranslation, 1, 1, TranslationRegime::El2, VaRange::Lower);
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[should_panic]
    fn no_el3_asid() {
        Mapping::new(IdTranslation, 1, 1, TranslationRegime::El3, VaRange::Lower);
    }
}
