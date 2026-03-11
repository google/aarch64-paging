// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! A library to manipulate AArch64 VMSA page tables.
//!
//! Currently it supports:
//!   - stage 1 and stage 2 page tables
//!   - 4 KiB pages
//!   - EL3, NS-EL2, NS-EL2&0 and NS-EL1&0 translation regimes
//!   - 64-bit descriptors
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
//!     descriptor::El1Attributes,
//!     paging::{MemoryRegion, El1And0},
//! };
//!
//! const ASID: usize = 1;
//! const ROOT_LEVEL: usize = 1;
//! const NORMAL_CACHEABLE: El1Attributes = El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE);
//!
//! // Create a new EL1 page table with identity mapping.
//! let mut idmap = IdMap::with_asid(ASID, ROOT_LEVEL, El1And0);
//! // Map a 2 MiB region of memory as read-write.
//! idmap.map_range(
//!     &MemoryRegion::new(0x80200000, 0x80400000),
//!     NORMAL_CACHEABLE | El1Attributes::NON_GLOBAL | El1Attributes::VALID | El1Attributes::ACCESSED,
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

pub mod descriptor;
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
use core::sync::atomic::{AtomicUsize, Ordering};
use descriptor::{
    Descriptor, DescriptorBits, PagingAttributes, PhysicalAddress, UpdatableDescriptor,
    VirtualAddress,
};
use paging::{Constraints, MemoryRegion, RootTable, Translation, TranslationRegime, VaRange};
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
    PteUpdateFault(DescriptorBits),
    /// The requested flags are not supported for this mapping
    #[error("Flags {0:#x} unsupported for mapping.")]
    InvalidFlags(usize),
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
pub struct Mapping<T: Translation<R::Attributes>, R: TranslationRegime> {
    root: RootTable<R, T>,
    asid: R::Asid,
    active_count: AtomicUsize,
}

/// Issues an inner-shareable data synchronization barrier (DSB) followed by an instruction
/// synchronization barrier (ISB) so that execution does not proceed until all TLB maintenance is
/// completed.
fn wait_for_tlb_maintenance() {
    // SAFETY: Barriers have no side effects that are observeable by the program
    #[cfg(target_arch = "aarch64")]
    unsafe {
        asm!("dsb ish", "isb", options(preserves_flags, nostack));
    }
}

impl<T: Translation<R::Attributes>, R: TranslationRegime<Asid = (), VaRange = ()>> Mapping<T, R> {
    /// Creates a new page table with the given root level and translation mapping.
    pub fn new(translation: T, rootlevel: usize, regime: R) -> Self {
        Self::new_impl(RootTable::new(translation, rootlevel, regime), ())
    }
}

impl<T: Translation<R::Attributes>, R: TranslationRegime<Asid = usize, VaRange = VaRange>>
    Mapping<T, R>
{
    /// Creates a new page table with the given ASID, root level and translation mapping.
    pub fn with_asid_and_va_range(
        translation: T,
        asid: usize,
        rootlevel: usize,
        regime: R,
        va_range: VaRange,
    ) -> Self {
        Self::new_impl(
            RootTable::with_va_range(translation, rootlevel, regime, va_range),
            asid,
        )
    }
}

impl<T: Translation<R::Attributes>, R: TranslationRegime> Mapping<T, R> {
    fn new_impl(root: RootTable<R, T>, asid: R::Asid) -> Self {
        Self {
            root,
            asid,
            active_count: AtomicUsize::new(0),
        }
    }

    /// Returns a reference to the translation used for this page table.
    pub fn translation(&self) -> &T {
        self.root.translation()
    }

    /// Returns whether this mapping is currently active.
    pub fn active(&self) -> bool {
        self.active_count.load(Ordering::Acquire) != 0
    }

    /// Returns the size in bytes of the virtual address space which can be mapped in this page
    /// table.
    ///
    /// This is a function of the chosen root level.
    pub fn size(&self) -> usize {
        self.root.size()
    }

    /// Activates the page table by programming the physical address of the root page table into
    /// `TTBRn_ELx`, along with the provided ASID. The previous value of `TTBRn_ELx` is returned so
    /// that it may later be restored by passing it to [`deactivate`](Self::deactivate).
    ///
    /// In test builds or builds that do not target aarch64, the `TTBRn_ELx` access is omitted.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the page table doesn't unmap any memory which the program is
    /// using, or introduce aliases which break Rust's aliasing rules. The page table must not be
    /// dropped while it is still active on any CPU.
    pub unsafe fn activate(&self) -> usize {
        #[allow(unused_mut, unused_assignments)]
        let mut previous_ttbr = usize::MAX;

        // Mark the page tables as active before actually activating them, to avoid a race
        // condition where a CPU observing the counter at zero might assume that the page tables
        // are not active yet, while they have already been loaded into the TTBR of another CPU.
        self.mark_active();

        #[cfg(all(not(test), target_arch = "aarch64"))]
        // SAFETY: We trust that self.root_address() returns a valid physical address of a page
        // table, and the `Drop` implementation will reset `TTBRn_ELx` before it becomes invalid.
        unsafe {
            // Ensure that all page table updates, as well as the increment of the active counter,
            // are visible to all observers before proceeding
            asm!("dmb ishst", "isb", options(preserves_flags),);
            previous_ttbr =
                R::activate(self.root_address(), self.asid, self.root.va_range_or_unit());
        }
        previous_ttbr
    }

    /// Deactivates the page table, by setting `TTBRn_ELx` to the provided value, and invalidating
    /// the TLB for this page table's configured ASID. The provided TTBR value should be the value
    /// returned by the preceding [`activate`](Self::activate) call.
    ///
    /// In test builds or builds that do not target aarch64, the `TTBRn_ELx` access is omitted.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the previous page table which this is switching back to doesn't
    /// unmap any memory which the program is using.
    pub unsafe fn deactivate(&self, #[allow(unused)] previous_ttbr: usize) {
        assert!(self.active());

        #[cfg(all(not(test), target_arch = "aarch64"))]
        // SAFETY: This just restores the previously saved value of `TTBRn_ELx`, which must have
        // been valid.
        unsafe {
            R::deactivate(previous_ttbr, self.asid, self.root.va_range_or_unit());
        }
        self.mark_inactive();
    }

    /// Checks whether the given range can be mapped or updated while the translation is live,
    /// without violating architectural break-before-make (BBM) requirements.
    fn check_range_bbm<F>(&self, range: &MemoryRegion, updater: &F) -> Result<(), MapError>
    where
        F: Fn(&MemoryRegion, &mut UpdatableDescriptor<R::Attributes>) -> Result<(), ()> + ?Sized,
    {
        self.root.visit_range(
            range,
            &mut |mr: &MemoryRegion, d: &Descriptor<R::Attributes>, level: usize| {
                let err = MapError::BreakBeforeMakeViolation(mr.clone());
                let mut desc = UpdatableDescriptor::clone_from(d, level);

                updater(mr, &mut desc)
                    .and_then(|_| {
                        if d.is_valid() && !mr.is_block(level) && d.flags() != desc.flags() {
                            // Cannot split a live block mapping
                            Err(())
                        } else {
                            Ok(())
                        }
                    })
                    .or(Err(err))
            },
        )
    }

    /// Invalidates `range` in the TLBs, so that permission changes are guaranteed to have taken
    /// effect by the time the function returns
    fn invalidate_range(&self, range: &MemoryRegion) {
        if self.active() {
            // If the mapping is active, no modifications are permitted that add or remove paging
            // levels. This means it is not necessary to iterate over the entire range at page
            // granularity, as invalidating a 2MiB block mapping or larger only requires a single
            // TLBI call.
            // If the mapping is not active, it was either never activated, or has previously been
            // deactivated, at which point TLB invalidation would have occurred, and so no TLB
            // maintenance is needed.
            self.root
                .visit_range(
                    range,
                    &mut |mr: &MemoryRegion, _: &Descriptor<R::Attributes>, _: usize| {
                        R::invalidate_va(mr.start());
                        Ok(())
                    },
                )
                .unwrap();

            wait_for_tlb_maintenance();
        }
    }

    /// Maps the given range of virtual addresses to the corresponding range of physical addresses
    /// starting at `pa`, with the given flags, taking the given constraints into account.
    ///
    /// To unmap a range, pass `flags` which don't contain the [`PagingAttributes::VALID`] bit.
    /// In this case the `pa` is ignored.
    ///
    /// This should generally only be called while the page table is not active. In particular, any
    /// change that may require break-before-make per the architecture must be made while the page
    /// table is inactive. Mapping a previously unmapped memory range may be done while the page
    /// table is active. This function writes block and page entries, but only maps them if `flags`
    /// contains [`PagingAttributes::VALID`], otherwise the entries remain invalid.
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
    /// Returns [`MapError::BreakBeforeMakeViolation`] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    pub fn map_range(
        &mut self,
        range: &MemoryRegion,
        pa: PhysicalAddress,
        flags: R::Attributes,
        constraints: Constraints,
    ) -> Result<(), MapError> {
        if self.active() {
            let c = |mr: &MemoryRegion, d: &mut UpdatableDescriptor<R::Attributes>| {
                let mask = !(paging::granularity_at_level(d.level()) - 1);
                let pa = (mr.start() - range.start() + pa.0) & mask;
                let flags = if d.level() == 3 {
                    flags | R::Attributes::TABLE_OR_PAGE
                } else {
                    flags
                };
                d.set(PhysicalAddress(pa), flags)
            };
            self.check_range_bbm(range, &c)?;
        }
        self.root.map_range(range, pa, flags, constraints)?;
        self.invalidate_range(range);
        Ok(())
    }

    /// Applies the provided updater function to a number of PTEs corresponding to a given memory range.
    ///
    /// This may involve splitting block entries if the provided range is not currently mapped
    /// down to its precise boundaries. For visiting all the descriptors covering a memory range
    /// without potential splitting (and no descriptor updates), use
    /// [`walk_range`](Self::walk_range) instead.
    ///
    /// The updater function receives the following arguments:
    ///
    /// - The virtual address range mapped by each page table descriptor. A new descriptor will
    ///   have been allocated before the invocation of the updater function if a page table split
    ///   was needed.
    /// - An `UpdatableDescriptor`, which includes a mutable reference to the page table descriptor
    ///   that permits modifications and the level of a translation table the descriptor belongs to.
    ///
    /// The updater function should return:
    ///
    /// - `Ok` to continue updating the remaining entries.
    /// - `Err` to signal an error and stop updating the remaining entries.
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
    /// Returns [`MapError::BreakBeforeMakeViolation`] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    pub fn modify_range<F>(&mut self, range: &MemoryRegion, f: &F) -> Result<(), MapError>
    where
        F: Fn(&MemoryRegion, &mut UpdatableDescriptor<R::Attributes>) -> Result<(), ()> + ?Sized,
    {
        if self.active() {
            self.check_range_bbm(range, f)?;
        }

        // modify_range() might fail halfway, in which case its Err() result will be returned
        // directly, and no barrier will be issued. The purpose of the barrier is to ensure that
        // the new state is visible to all observers before proceeding, but in case of a failure,
        // what that new state entails is uncertain anyway, and so there is no point in
        // synchronizing it.
        if self.root.modify_range(range, f, self.active())? && self.active() {
            wait_for_tlb_maintenance();
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
        F: FnMut(&MemoryRegion, &Descriptor<R::Attributes>, usize) -> Result<(), ()>,
    {
        self.root.walk_range(range, f)
    }

    /// Looks for subtables whose entries are all empty and replaces them with a single empty entry,
    /// freeing the subtable.
    ///
    /// This requires walking the whole hierarchy of pagetables, so you may not want to call it
    /// every time a region is unmapped. You could instead call it when the system is under memory
    /// pressure.
    pub fn compact_subtables(&mut self) {
        self.root.compact_subtables();
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
    pub fn asid(&self) -> R::Asid {
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
    pub fn mark_active(&self) {
        self.active_count.fetch_add(1, Ordering::Release);
    }

    /// Marks the page table as inactive.
    ///
    /// This may be called after manually disabling the use of the page table, such as by setting
    /// the relevant TTBR to a different address.
    ///
    /// It is called automatically by [`deactivate`](Self::deactivate).
    pub fn mark_inactive(&self) {
        let l = self.active_count.fetch_sub(1, Ordering::Release);
        if l == 0 {
            // If the old value was 0, the new value underflowed
            panic!("Underflow in active count.");
        }
    }
}

impl<T: Translation<R::Attributes>, R: TranslationRegime> Drop for Mapping<T, R> {
    fn drop(&mut self) {
        if self.active() {
            panic!("Dropping active page table mapping!");
        }
    }
}
