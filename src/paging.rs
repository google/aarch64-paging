// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Generic aarch64 page table manipulation functionality which doesn't assume anything about how
//! addresses are mapped.

use crate::MapError;
use crate::descriptor::{
    Attributes, Descriptor, PhysicalAddress, UpdatableDescriptor, VirtualAddress,
};

#[cfg(feature = "alloc")]
use alloc::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use bitflags::bitflags;
#[cfg(all(not(test), target_arch = "aarch64"))]
use core::arch::asm;
use core::fmt::{self, Debug, Display, Formatter};
use core::marker::PhantomData;
use core::ops::Range;
use core::ptr::NonNull;
use core::sync::atomic::AtomicUsize;

const PAGE_SHIFT: usize = 12;

/// The pagetable level at which all entries are page mappings.
pub const LEAF_LEVEL: usize = 3;

/// The page size in bytes assumed by this library, 4 KiB.
pub const PAGE_SIZE: usize = 1 << PAGE_SHIFT;

/// The number of address bits resolved in one level of page table lookup. This is a function of the
/// page size.
pub const BITS_PER_LEVEL: usize = PAGE_SHIFT - 3;

/// Which virtual address range a page table is for, i.e. which TTBR register to use for it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VaRange {
    /// The page table covers the bottom of the virtual address space (starting at address 0), so
    /// will be used with `TTBR0`.
    Lower,
    /// The page table covers the top of the virtual address space (ending at address
    /// 0xffff_ffff_ffff_ffff), so will be used with `TTBR1`.
    Upper,
}

/// Which translation regime a page table is for.
///
/// This depends on the exception level, among other things.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TranslationRegime {
    /// Secure EL3.
    El3,
    /// Non-secure EL2.
    El2,
    /// Non-secure EL2&0, with VHE.
    El2And0,
    /// Non-secure EL1&0, stage 1.
    El1And0,
}

impl TranslationRegime {
    /// Returns whether this translation regime supports use of an ASID.
    ///
    /// This also implies that it supports two VA ranges.
    pub(crate) fn supports_asid(self) -> bool {
        matches!(self, Self::El2And0 | Self::El1And0)
    }

    /// Invalidates the memory range starting at `va`. The size of the range is unspecified, it is
    /// up to the caller to apply this function as needed to invalidate a range of memory
    pub(crate) fn invalidate_va(self, va: VirtualAddress) {
        #[allow(unused)]
        let va = va.0 >> 12;

        // SAFETY: TLBI maintenance has no side effects that are observeable by the
        // program
        #[cfg(all(not(test), target_arch = "aarch64"))]
        unsafe {
            match self {
                TranslationRegime::El3 => asm!(
                    "tlbi vae3is, {va}",
                    va = in(reg) va,
                    options(preserves_flags, nostack),
                ),
                TranslationRegime::El2 | TranslationRegime::El2And0 => asm!(
                    "tlbi vae2is, {va}",
                    va = in(reg) va,
                    options(preserves_flags, nostack),
                ),
                TranslationRegime::El1And0 => asm!(
                    "tlbi vaae1is, {va}",
                    va = in(reg) va,
                    options(preserves_flags, nostack),
                ),
            };
        };
    }
}

/// A range of virtual addresses which may be mapped in a page table.
#[derive(Clone, Eq, PartialEq)]
pub struct MemoryRegion(Range<VirtualAddress>);

/// Returns the size in bytes of the address space covered by a single entry in the page table at
/// the given level.
pub(crate) fn granularity_at_level(level: usize) -> usize {
    PAGE_SIZE << ((LEAF_LEVEL - level) * BITS_PER_LEVEL)
}

/// An implementation of this trait needs to be provided to the mapping routines, so that the
/// physical addresses used in the page tables can be converted into virtual addresses that can be
/// used to access their contents from the code.
pub trait Translation {
    /// Allocates a zeroed page, which is already mapped, to be used for a new subtable of some
    /// pagetable. Returns both a pointer to the page and its physical address.
    fn allocate_table(&mut self) -> (NonNull<PageTable>, PhysicalAddress);

    /// Deallocates the page which was previous allocated by [`allocate_table`](Self::allocate_table).
    ///
    /// # Safety
    ///
    /// The memory must have been allocated by `allocate_table` on the same `Translation`, and not
    /// yet deallocated.
    unsafe fn deallocate_table(&mut self, page_table: NonNull<PageTable>);

    /// Given the physical address of a subtable, returns the virtual address at which it is mapped.
    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable>;
}

impl MemoryRegion {
    /// Constructs a new `MemoryRegion` for the given range of virtual addresses.
    ///
    /// The start is inclusive and the end is exclusive. Both will be aligned to the [`PAGE_SIZE`],
    /// with the start being rounded down and the end being rounded up.
    pub const fn new(start: usize, end: usize) -> MemoryRegion {
        MemoryRegion(
            VirtualAddress(align_down(start, PAGE_SIZE))..VirtualAddress(align_up(end, PAGE_SIZE)),
        )
    }

    /// Returns the first virtual address of the memory range.
    pub const fn start(&self) -> VirtualAddress {
        self.0.start
    }

    /// Returns the first virtual address after the memory range.
    pub const fn end(&self) -> VirtualAddress {
        self.0.end
    }

    /// Returns the length of the memory region in bytes.
    pub const fn len(&self) -> usize {
        self.0.end.0 - self.0.start.0
    }

    /// Returns whether the memory region contains exactly 0 bytes.
    pub const fn is_empty(&self) -> bool {
        self.0.start.0 == self.0.end.0
    }

    fn split(&self, level: usize) -> ChunkedIterator<'_> {
        ChunkedIterator {
            range: self,
            granularity: granularity_at_level(level),
            start: self.0.start.0,
        }
    }

    /// Returns whether this region can be mapped at 'level' using block mappings only.
    pub(crate) fn is_block(&self, level: usize) -> bool {
        let gran = granularity_at_level(level);
        (self.0.start.0 | self.0.end.0) & (gran - 1) == 0
    }
}

impl From<Range<VirtualAddress>> for MemoryRegion {
    fn from(range: Range<VirtualAddress>) -> Self {
        Self::new(range.start.0, range.end.0)
    }
}

impl Display for MemoryRegion {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}..{}", self.0.start, self.0.end)
    }
}

impl Debug for MemoryRegion {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        Display::fmt(self, f)
    }
}

bitflags! {
    /// Constraints on page table mappings
    #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct Constraints: usize {
        /// Block mappings are not permitted, only page mappings
        const NO_BLOCK_MAPPINGS    = 1 << 0;
        /// Use of the contiguous hint is not permitted
        const NO_CONTIGUOUS_HINT   = 1 << 1;
    }
}

/// A complete hierarchy of page tables including all levels.
pub struct RootTable<T: Translation> {
    table: PageTableWithLevel<T>,
    translation: T,
    pa: PhysicalAddress,
    translation_regime: TranslationRegime,
    va_range: VaRange,
}

impl<T: Translation> RootTable<T> {
    /// Creates a new page table starting at the given root level.
    ///
    /// The level must be between 0 and 3; level -1 (for 52-bit addresses with LPA2) is not
    /// currently supported by this library. The value of `TCR_EL1.T0SZ` must be set appropriately
    /// to match.
    pub fn new(
        mut translation: T,
        level: usize,
        translation_regime: TranslationRegime,
        va_range: VaRange,
    ) -> Self {
        if level > LEAF_LEVEL {
            panic!("Invalid root table level {}.", level);
        }
        if !translation_regime.supports_asid() && va_range != VaRange::Lower {
            panic!(
                "{:?} doesn't have an upper virtual address range.",
                translation_regime
            );
        }
        let (table, pa) = PageTableWithLevel::new(&mut translation, level);
        RootTable {
            table,
            translation,
            pa,
            translation_regime,
            va_range,
        }
    }

    /// Returns the size in bytes of the virtual address space which can be mapped in this page
    /// table.
    ///
    /// This is a function of the chosen root level.
    pub fn size(&self) -> usize {
        granularity_at_level(self.table.level) << BITS_PER_LEVEL
    }

    /// Recursively maps a range into the pagetable hierarchy starting at the root level, mapping
    /// the pages to the corresponding physical address range starting at `pa`. Block and page
    /// entries will be written to, but will only be mapped if `flags` contains `Attributes::VALID`.
    ///
    /// Returns an error if the virtual address range is out of the range covered by the pagetable,
    /// or if the `flags` argument has unsupported attributes set.
    pub fn map_range(
        &mut self,
        range: &MemoryRegion,
        pa: PhysicalAddress,
        flags: Attributes,
        constraints: Constraints,
    ) -> Result<(), MapError> {
        if flags.contains(Attributes::TABLE_OR_PAGE) {
            return Err(MapError::InvalidFlags(Attributes::TABLE_OR_PAGE));
        }
        self.verify_region(range)?;
        self.table
            .map_range(&mut self.translation, range, pa, flags, constraints);
        Ok(())
    }

    /// Returns the physical address of the root table in memory.
    pub fn to_physical(&self) -> PhysicalAddress {
        self.pa
    }

    /// Returns the virtual address range for which this table is intended.
    ///
    /// This affects which TTBR register is used.
    pub fn va_range(&self) -> VaRange {
        self.va_range
    }

    /// Returns the translation regime for which this table is intended.
    pub fn translation_regime(&self) -> TranslationRegime {
        self.translation_regime
    }

    /// Returns a reference to the translation used for this page table.
    pub fn translation(&self) -> &T {
        &self.translation
    }

    /// Applies the provided updater function to the page table descriptors covering a given
    /// memory range.
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
    /// - A mutable reference to the page table descriptor that permits modifications.
    /// - The level of a translation table the descriptor belongs to.
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
    pub(crate) fn modify_range<F>(
        &mut self,
        range: &MemoryRegion,
        f: &F,
        live: bool,
    ) -> Result<bool, MapError>
    where
        F: Fn(&MemoryRegion, &mut UpdatableDescriptor) -> Result<(), ()> + ?Sized,
    {
        self.verify_region(range)?;
        self.table.modify_range(
            &mut self.translation,
            self.translation_regime,
            range,
            f,
            live,
        )
    }

    /// Applies the provided callback function to the page table descriptors covering a given
    /// memory range.
    ///
    /// The callback function receives the following arguments:
    ///
    /// - The range covered by the current step in the walk. This is always a subrange of `range`
    ///   even when the descriptor covers a region that exceeds it.
    /// - The page table descriptor itself.
    /// - The level of a translation table the descriptor belongs to.
    ///
    /// The callback function should return:
    ///
    /// - `Ok` to continue visiting the remaining entries.
    /// - `Err` to signal an error and stop visiting the remaining entries.
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
        self.visit_range(range, &mut |mr, desc, level| {
            f(mr, desc, level).map_err(|_| MapError::PteUpdateFault(desc.bits()))
        })
    }

    // Private version of `walk_range` using a closure that returns MapError on error
    pub(crate) fn visit_range<F>(&self, range: &MemoryRegion, f: &mut F) -> Result<(), MapError>
    where
        F: FnMut(&MemoryRegion, &Descriptor, usize) -> Result<(), MapError>,
    {
        self.verify_region(range)?;
        self.table.visit_range(&self.translation, range, f)
    }

    /// Returns the level of mapping used for the given virtual address:
    /// - `None` if it is unmapped
    /// - `Some(LEAF_LEVEL)` if it is mapped as a single page
    /// - `Some(level)` if it is mapped as a block at `level`
    #[cfg(all(test, feature = "alloc"))]
    pub(crate) fn mapping_level(&self, va: VirtualAddress) -> Option<usize> {
        self.table.mapping_level(&self.translation, va)
    }

    /// Checks whether the region is within range of the page table.
    fn verify_region(&self, region: &MemoryRegion) -> Result<(), MapError> {
        if region.end() < region.start() {
            return Err(MapError::RegionBackwards(region.clone()));
        }
        match self.va_range {
            VaRange::Lower => {
                if (region.start().0 as isize) < 0 {
                    return Err(MapError::AddressRange(region.start()));
                } else if region.end().0 > self.size() {
                    return Err(MapError::AddressRange(region.end()));
                }
            }
            VaRange::Upper => {
                if region.start().0 as isize >= 0
                    || (region.start().0 as isize).unsigned_abs() > self.size()
                {
                    return Err(MapError::AddressRange(region.start()));
                }
            }
        }
        Ok(())
    }
}

impl<T: Translation> Debug for RootTable<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        writeln!(
            f,
            "RootTable {{ pa: {}, translation_regime: {:?}, va_range: {:?}, level: {}, table:",
            self.pa, self.translation_regime, self.va_range, self.table.level
        )?;
        self.table.fmt_indented(f, &self.translation, 0)?;
        write!(f, "}}")
    }
}

impl<T: Translation> Drop for RootTable<T> {
    fn drop(&mut self) {
        // SAFETY: We created the table in `RootTable::new` by calling `PageTableWithLevel::new`
        // with `self.translation`. Subtables were similarly created by
        // `PageTableWithLevel::split_entry` calling `PageTableWithLevel::new` with the same
        // translation.
        unsafe { self.table.free(&mut self.translation) }
    }
}

struct ChunkedIterator<'a> {
    range: &'a MemoryRegion,
    granularity: usize,
    start: usize,
}

impl Iterator for ChunkedIterator<'_> {
    type Item = MemoryRegion;

    fn next(&mut self) -> Option<MemoryRegion> {
        if !self.range.0.contains(&VirtualAddress(self.start)) {
            return None;
        }
        let end = self
            .range
            .0
            .end
            .0
            .min((self.start | (self.granularity - 1)) + 1);
        let c = MemoryRegion::new(self.start, end);
        self.start = end;
        Some(c)
    }
}

/// Smart pointer which owns a [`PageTable`] and knows what level it is at. This allows it to
/// implement `Debug` and `Drop`, as walking the page table hierachy requires knowing the starting
/// level.
#[derive(Debug)]
pub(crate) struct PageTableWithLevel<T: Translation> {
    table: NonNull<PageTable>,
    level: usize,
    _translation: PhantomData<T>,
}

// SAFETY: The underlying PageTable is process-wide and can be safely accessed from any thread
// with appropriate synchronization. This type manages ownership for the raw pointer.
unsafe impl<T: Translation + Send> Send for PageTableWithLevel<T> {}

// SAFETY: &Self only allows reading from the page table, which is safe to do from any thread.
unsafe impl<T: Translation + Sync> Sync for PageTableWithLevel<T> {}

impl<T: Translation> PageTableWithLevel<T> {
    /// Allocates a new, zeroed, appropriately-aligned page table with the given translation,
    /// returning both a pointer to it and its physical address.
    fn new(translation: &mut T, level: usize) -> (Self, PhysicalAddress) {
        assert!(level <= LEAF_LEVEL);
        let (table, pa) = translation.allocate_table();
        (
            // Safe because the pointer has been allocated with the appropriate layout, and the
            // memory is zeroed which is valid initialisation for a PageTable.
            Self::from_pointer(table, level),
            pa,
        )
    }

    pub(crate) fn from_pointer(table: NonNull<PageTable>, level: usize) -> Self {
        Self {
            table,
            level,
            _translation: PhantomData,
        }
    }

    /// Returns a reference to the descriptor corresponding to a given virtual address.
    fn get_entry(&self, va: VirtualAddress) -> &Descriptor {
        let shift = PAGE_SHIFT + (LEAF_LEVEL - self.level) * BITS_PER_LEVEL;
        let index = (va.0 >> shift) % (1 << BITS_PER_LEVEL);
        // SAFETY: We know that the pointer is properly aligned, dereferenced and initialised, and
        // nothing else can access the page table while we hold a mutable reference to the
        // PageTableWithLevel (assuming it is not currently active).
        let table = unsafe { self.table.as_ref() };
        &table.entries[index]
    }

    /// Returns a mutable reference to the descriptor corresponding to a given virtual address.
    fn get_entry_mut(&mut self, va: VirtualAddress) -> &mut Descriptor {
        let shift = PAGE_SHIFT + (LEAF_LEVEL - self.level) * BITS_PER_LEVEL;
        let index = (va.0 >> shift) % (1 << BITS_PER_LEVEL);
        // SAFETY: We know that the pointer is properly aligned, dereferenced and initialised, and
        // nothing else can access the page table while we hold a mutable reference to the
        // PageTableWithLevel (assuming it is not currently active).
        let table = unsafe { self.table.as_mut() };
        &mut table.entries[index]
    }

    /// Convert the descriptor in `entry` from a block mapping to a table mapping of
    /// the same range with the same attributes
    fn split_entry(
        translation: &mut T,
        chunk: &MemoryRegion,
        entry: &mut Descriptor,
        level: usize,
    ) -> Self {
        let granularity = granularity_at_level(level);
        let (mut subtable, subtable_pa) = Self::new(translation, level + 1);
        let old_flags = entry.flags();
        let old_pa = entry.output_address();
        if !old_flags.contains(Attributes::TABLE_OR_PAGE)
            && (!old_flags.is_empty() || old_pa.0 != 0)
        {
            // `old` was a block entry, so we need to split it.
            // Recreate the entire block in the newly added table.
            let a = align_down(chunk.0.start.0, granularity);
            let b = align_up(chunk.0.end.0, granularity);
            subtable.map_range(
                translation,
                &MemoryRegion::new(a, b),
                old_pa,
                old_flags,
                Constraints::empty(),
            );
        }
        // If `old` was not a block entry, a newly zeroed page will be added to the hierarchy,
        // which might be live in this case. We rely on the release semantics of the set() below to
        // ensure that all observers that see the new entry will also see the zeroed contents.
        entry.set(subtable_pa, Attributes::TABLE_OR_PAGE | Attributes::VALID);
        subtable
    }

    /// Maps the the given virtual address range in this pagetable to the corresponding physical
    /// address range starting at the given `pa`, recursing into any subtables as necessary. To map
    /// block and page entries, `Attributes::VALID` must be set in `flags`.
    ///
    /// Assumes that the entire range is within the range covered by this pagetable.
    ///
    /// Panics if the `translation` doesn't provide a corresponding physical address for some
    /// virtual address within the range, as there is no way to roll back to a safe state so this
    /// should be checked by the caller beforehand.
    fn map_range(
        &mut self,
        translation: &mut T,
        range: &MemoryRegion,
        mut pa: PhysicalAddress,
        flags: Attributes,
        constraints: Constraints,
    ) {
        let level = self.level;
        let granularity = granularity_at_level(level);

        for chunk in range.split(level) {
            let entry = self.get_entry_mut(chunk.0.start);

            if level == LEAF_LEVEL {
                // Put down a page mapping.
                entry.set(pa, flags | Attributes::TABLE_OR_PAGE);
            } else if !entry.is_table_or_page()
                && entry.flags() == flags
                && entry.output_address().0 == pa.0 - chunk.0.start.0 % granularity
            {
                // There is no need to split up a block mapping if it already maps the desired `pa`
                // with the desired `flags`. So do nothing in this case.
            } else if chunk.is_block(level)
                && !entry.is_table_or_page()
                && is_aligned(pa.0, granularity)
                && !constraints.contains(Constraints::NO_BLOCK_MAPPINGS)
            {
                // Rather than leak the entire subhierarchy, only put down
                // a block mapping if the region is not already covered by
                // a table mapping.
                entry.set(pa, flags);
            } else {
                let mut subtable = entry
                    .subtable(translation, level)
                    .unwrap_or_else(|| Self::split_entry(translation, &chunk, entry, level));
                subtable.map_range(translation, &chunk, pa, flags, constraints);
            }
            pa.0 += chunk.len();
        }
    }

    fn fmt_indented(
        &self,
        f: &mut Formatter,
        translation: &T,
        indentation: usize,
    ) -> Result<(), fmt::Error> {
        const WIDTH: usize = 3;
        // SAFETY: We know that the pointer is aligned, initialised and dereferencable, and the
        // PageTable won't be mutated while we are using it.
        let table = unsafe { self.table.as_ref() };

        let mut i = 0;
        while i < table.entries.len() {
            if let Some(subtable) = table.entries[i].subtable(translation, self.level) {
                writeln!(
                    f,
                    "{:indentation$}{: <WIDTH$}    : {:?}",
                    "", i, table.entries[i],
                )?;
                subtable.fmt_indented(f, translation, indentation + 2)?;
                i += 1;
            } else {
                let first_contiguous = i;
                let first_entry = table.entries[i].bits();
                let granularity = granularity_at_level(self.level);
                while i < table.entries.len()
                    && (table.entries[i].bits() == first_entry
                        || (first_entry != 0
                            && table.entries[i].bits()
                                == first_entry + granularity * (i - first_contiguous)))
                {
                    i += 1;
                }
                if i - 1 == first_contiguous {
                    write!(f, "{:indentation$}{: <WIDTH$}    : ", "", first_contiguous)?;
                } else {
                    write!(
                        f,
                        "{:indentation$}{: <WIDTH$}-{: <WIDTH$}: ",
                        "",
                        first_contiguous,
                        i - 1,
                    )?;
                }
                if first_entry == 0 {
                    writeln!(f, "0")?;
                } else {
                    writeln!(f, "{:?}", Descriptor(AtomicUsize::new(first_entry)))?;
                }
            }
        }
        Ok(())
    }

    /// Frees the memory used by this pagetable and all subtables. It is not valid to access the
    /// page table after this.
    ///
    /// # Safety
    ///
    /// The table and all its subtables must have been created by `PageTableWithLevel::new` with the
    /// same `translation`.
    unsafe fn free(&mut self, translation: &mut T) {
        // SAFETY: We know that the pointer is aligned, initialised and dereferencable, and the
        // PageTable won't be mutated while we are freeing it.
        let table = unsafe { self.table.as_ref() };
        for entry in &table.entries {
            if let Some(mut subtable) = entry.subtable(translation, self.level) {
                // SAFETY: Our caller promised that all our subtables were created by
                // `PageTableWithLevel::new` with the same `translation`.
                unsafe {
                    subtable.free(translation);
                }
            }
        }
        // SAFETY: Our caller promised that the table was created by `PageTableWithLevel::new` with
        // `translation`, which then allocated it by calling `allocate_table` on `translation`.
        unsafe {
            // Actually free the memory used by the `PageTable`.
            translation.deallocate_table(self.table);
        }
    }

    /// Modifies a range of page table entries by applying a function to each page table entry.
    /// If the range is not aligned to block boundaries, block descriptors will be split up.
    fn modify_range<F>(
        &mut self,
        translation: &mut T,
        translation_regime: TranslationRegime,
        range: &MemoryRegion,
        f: &F,
        live: bool,
    ) -> Result<bool, MapError>
    where
        F: Fn(&MemoryRegion, &mut UpdatableDescriptor) -> Result<(), ()> + ?Sized,
    {
        let mut modified = false;
        let level = self.level;
        for chunk in range.split(level) {
            let entry = self.get_entry_mut(chunk.0.start);
            if let Some(mut subtable) = entry.subtable(translation, level).or_else(|| {
                if !chunk.is_block(level) {
                    // The current chunk is not aligned to the block size at this level
                    // Split it before recursing to the next level
                    Some(Self::split_entry(translation, &chunk, entry, level))
                } else {
                    None
                }
            }) {
                modified |=
                    subtable.modify_range(translation, translation_regime, &chunk, f, live)?;
            } else {
                let bits = entry.bits();
                let mut desc = UpdatableDescriptor::new(entry, level, live);
                f(&chunk, &mut desc).map_err(|_| MapError::PteUpdateFault(bits))?;

                if live && desc.updated() {
                    // Live descriptor was updated so TLB maintenance is needed
                    translation_regime.invalidate_va(chunk.start());
                    modified = true;
                }
            }
        }
        Ok(modified)
    }

    /// Walks a range of page table entries and passes each one to a caller provided function.
    /// If the function returns an error, the walk is terminated and the error value is passed on
    fn visit_range<F, E>(&self, translation: &T, range: &MemoryRegion, f: &mut F) -> Result<(), E>
    where
        F: FnMut(&MemoryRegion, &Descriptor, usize) -> Result<(), E>,
    {
        let level = self.level;
        for chunk in range.split(level) {
            let entry = self.get_entry(chunk.0.start);
            if let Some(subtable) = entry.subtable(translation, level) {
                subtable.visit_range(translation, &chunk, f)?;
            } else {
                f(&chunk, entry, level)?;
            }
        }
        Ok(())
    }

    /// Returns the level of mapping used for the given virtual address:
    /// - `None` if it is unmapped
    /// - `Some(LEAF_LEVEL)` if it is mapped as a single page
    /// - `Some(level)` if it is mapped as a block at `level`
    #[cfg(all(test, feature = "alloc"))]
    fn mapping_level(&self, translation: &T, va: VirtualAddress) -> Option<usize> {
        let entry = self.get_entry(va);
        if let Some(subtable) = entry.subtable(translation, self.level) {
            subtable.mapping_level(translation, va)
        } else {
            if entry.is_valid() {
                Some(self.level)
            } else {
                None
            }
        }
    }
}

/// A single level of a page table.
#[repr(C, align(4096))]
pub struct PageTable {
    entries: [Descriptor; 1 << BITS_PER_LEVEL],
}

impl PageTable {
    /// An empty (i.e. zeroed) page table. This may be useful for initialising statics.
    pub const EMPTY: Self = Self {
        entries: [Descriptor::EMPTY; 1 << BITS_PER_LEVEL],
    };

    /// Allocates a new zeroed, appropriately-aligned pagetable on the heap using the global
    /// allocator and returns a pointer to it.
    #[cfg(feature = "alloc")]
    pub fn new() -> NonNull<Self> {
        // SAFETY: Zeroed memory is a valid initialisation for a PageTable.
        unsafe { allocate_zeroed() }
    }

    /// Write the in-memory presentation of the page table to the byte slice referenced by `page`.
    ///
    /// Returns `Ok(())` on success, or `Err(())` if the size of the byte slice is not equal to the
    /// size of a page table.
    pub fn write_to(&self, page: &mut [u8]) -> Result<(), ()> {
        if page.len() != self.entries.len() * size_of::<Descriptor>() {
            return Err(());
        }
        for (chunk, desc) in page
            .chunks_exact_mut(size_of::<Descriptor>())
            .zip(self.entries.iter())
        {
            chunk.copy_from_slice(&desc.bits().to_le_bytes());
        }
        Ok(())
    }
}

impl Default for PageTable {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Allocates appropriately aligned heap space for a `T` and zeroes it.
///
/// # Safety
///
/// It must be valid to initialise the type `T` by simply zeroing its memory.
#[cfg(feature = "alloc")]
unsafe fn allocate_zeroed<T>() -> NonNull<T> {
    let layout = Layout::new::<T>();
    assert_ne!(layout.size(), 0);
    // SAFETY: We just checked that the layout has non-zero size.
    let pointer = unsafe { alloc_zeroed(layout) };
    if pointer.is_null() {
        handle_alloc_error(layout);
    }
    // SAFETY: We just checked that the pointer is non-null.
    unsafe { NonNull::new_unchecked(pointer as *mut T) }
}

/// Deallocates the heap space for a `T` which was previously allocated by `allocate_zeroed`.
///
/// # Safety
///
/// The memory must have been allocated by the global allocator, with the layout for `T`, and not
/// yet deallocated.
#[cfg(feature = "alloc")]
pub(crate) unsafe fn deallocate<T>(ptr: NonNull<T>) {
    let layout = Layout::new::<T>();
    // SAFETY: We delegate the safety requirements to our caller.
    unsafe {
        dealloc(ptr.as_ptr() as *mut u8, layout);
    }
}

const fn align_down(value: usize, alignment: usize) -> usize {
    value & !(alignment - 1)
}

const fn align_up(value: usize, alignment: usize) -> usize {
    ((value - 1) | (alignment - 1)) + 1
}

pub(crate) const fn is_aligned(value: usize, alignment: usize) -> bool {
    value & (alignment - 1) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "alloc")]
    use crate::idmap::IdTranslation;
    #[cfg(feature = "alloc")]
    use crate::target::TargetAllocator;
    #[cfg(feature = "alloc")]
    use alloc::{format, string::ToString, vec, vec::Vec};
    use core::sync::atomic::AtomicUsize;

    #[cfg(feature = "alloc")]
    #[test]
    fn display_memory_region() {
        let region = MemoryRegion::new(0x1234, 0x56789);
        assert_eq!(
            &region.to_string(),
            "0x0000000000001000..0x0000000000057000"
        );
        assert_eq!(
            &format!("{:?}", region),
            "0x0000000000001000..0x0000000000057000"
        );
    }

    #[test]
    fn subtract_virtual_address() {
        let low = VirtualAddress(0x12);
        let high = VirtualAddress(0x1234);
        assert_eq!(high - low, 0x1222);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn subtract_virtual_address_overflow() {
        let low = VirtualAddress(0x12);
        let high = VirtualAddress(0x1234);

        // This would overflow, so should panic.
        let _ = low - high;
    }

    #[test]
    fn add_virtual_address() {
        assert_eq!(VirtualAddress(0x1234) + 0x42, VirtualAddress(0x1276));
    }

    #[test]
    fn subtract_physical_address() {
        let low = PhysicalAddress(0x12);
        let high = PhysicalAddress(0x1234);
        assert_eq!(high - low, 0x1222);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn subtract_physical_address_overflow() {
        let low = PhysicalAddress(0x12);
        let high = PhysicalAddress(0x1234);

        // This would overflow, so should panic.
        let _ = low - high;
    }

    #[test]
    fn add_physical_address() {
        assert_eq!(PhysicalAddress(0x1234) + 0x42, PhysicalAddress(0x1276));
    }

    #[test]
    fn invalid_descriptor() {
        let desc = Descriptor(AtomicUsize::new(0usize));
        assert!(!desc.is_valid());
        assert!(!desc.flags().contains(Attributes::VALID));
    }

    #[test]
    fn set_descriptor() {
        const PHYSICAL_ADDRESS: usize = 0x12340000;
        let mut desc = Descriptor(AtomicUsize::new(0usize));
        assert!(!desc.is_valid());
        desc.set(
            PhysicalAddress(PHYSICAL_ADDRESS),
            Attributes::TABLE_OR_PAGE | Attributes::USER | Attributes::SWFLAG_1 | Attributes::VALID,
        );
        assert!(desc.is_valid());
        assert_eq!(
            desc.flags(),
            Attributes::TABLE_OR_PAGE | Attributes::USER | Attributes::SWFLAG_1 | Attributes::VALID
        );
        assert_eq!(desc.output_address(), PhysicalAddress(PHYSICAL_ADDRESS));
    }

    #[test]
    fn modify_descriptor_flags() {
        let mut desc = Descriptor(AtomicUsize::new(0usize));
        assert!(!desc.is_valid());
        desc.set(
            PhysicalAddress(0x12340000),
            Attributes::TABLE_OR_PAGE | Attributes::USER | Attributes::SWFLAG_1,
        );
        UpdatableDescriptor::new(&mut desc, 3, true)
            .modify_flags(
                Attributes::DBM | Attributes::SWFLAG_3,
                Attributes::VALID | Attributes::SWFLAG_1,
            )
            .unwrap();
        assert!(!desc.is_valid());
        assert_eq!(
            desc.flags(),
            Attributes::TABLE_OR_PAGE | Attributes::USER | Attributes::SWFLAG_3 | Attributes::DBM
        );
    }

    #[test]
    #[should_panic]
    fn modify_descriptor_table_or_page_flag() {
        let mut desc = Descriptor(AtomicUsize::new(0usize));
        assert!(!desc.is_valid());
        desc.set(
            PhysicalAddress(0x12340000),
            Attributes::TABLE_OR_PAGE | Attributes::USER | Attributes::SWFLAG_1,
        );
        UpdatableDescriptor::new(&mut desc, 3, false)
            .modify_flags(Attributes::VALID, Attributes::TABLE_OR_PAGE)
            .unwrap();
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn unaligned_chunks() {
        let region = MemoryRegion::new(0x0000_2000, 0x0020_5000);
        let chunks = region.split(LEAF_LEVEL - 1).collect::<Vec<_>>();
        assert_eq!(
            chunks,
            vec![
                MemoryRegion::new(0x0000_2000, 0x0020_0000),
                MemoryRegion::new(0x0020_0000, 0x0020_5000),
            ]
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[should_panic]
    fn no_el2_ttbr1() {
        RootTable::<IdTranslation>::new(IdTranslation, 1, TranslationRegime::El2, VaRange::Upper);
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[should_panic]
    fn no_el3_ttbr1() {
        RootTable::<IdTranslation>::new(IdTranslation, 1, TranslationRegime::El3, VaRange::Upper);
    }

    #[test]
    fn table_or_page() {
        // Invalid.
        assert!(!Descriptor(AtomicUsize::new(0b00)).is_table_or_page());
        assert!(!Descriptor(AtomicUsize::new(0b10)).is_table_or_page());

        // Block mapping.
        assert!(!Descriptor(AtomicUsize::new(0b01)).is_table_or_page());

        // Table or page.
        assert!(Descriptor(AtomicUsize::new(0b11)).is_table_or_page());
    }

    #[test]
    fn table_or_page_unknown_bits() {
        // Some RES0 and IGNORED bits that we set for the sake of the test.
        const UNKNOWN: usize = 1 << 50 | 1 << 52;

        // Invalid.
        assert!(!Descriptor(AtomicUsize::new(UNKNOWN | 0b00)).is_table_or_page());
        assert!(!Descriptor(AtomicUsize::new(UNKNOWN | 0b10)).is_table_or_page());

        // Block mapping.
        assert!(!Descriptor(AtomicUsize::new(UNKNOWN | 0b01)).is_table_or_page());

        // Table or page.
        assert!(Descriptor(AtomicUsize::new(UNKNOWN | 0b11)).is_table_or_page());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn debug_roottable_empty() {
        let table = RootTable::<TargetAllocator>::new(
            TargetAllocator::new(0),
            1,
            TranslationRegime::El1And0,
            VaRange::Lower,
        );
        assert_eq!(
            format!("{table:?}"),
"RootTable { pa: 0x0000000000000000, translation_regime: El1And0, va_range: Lower, level: 1, table:
0  -511: 0
}"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn debug_roottable_contiguous() {
        let mut table = RootTable::<TargetAllocator>::new(
            TargetAllocator::new(0),
            1,
            TranslationRegime::El1And0,
            VaRange::Lower,
        );
        table
            .map_range(
                &MemoryRegion::new(PAGE_SIZE * 3, PAGE_SIZE * 6),
                PhysicalAddress(PAGE_SIZE * 3),
                Attributes::VALID | Attributes::NON_GLOBAL,
                Constraints::empty(),
            )
            .unwrap();
        table
            .map_range(
                &MemoryRegion::new(PAGE_SIZE * 6, PAGE_SIZE * 7),
                PhysicalAddress(PAGE_SIZE * 6),
                Attributes::VALID | Attributes::READ_ONLY,
                Constraints::empty(),
            )
            .unwrap();
        table
            .map_range(
                &MemoryRegion::new(PAGE_SIZE * 8, PAGE_SIZE * 9),
                PhysicalAddress(PAGE_SIZE * 8),
                Attributes::VALID | Attributes::READ_ONLY,
                Constraints::empty(),
            )
            .unwrap();
        assert_eq!(
            format!("{table:?}"),
"RootTable { pa: 0x0000000000000000, translation_regime: El1And0, va_range: Lower, level: 1, table:
0      : 0x00000000001003 (0x0000000000001000, Attributes(VALID | TABLE_OR_PAGE))
  0      : 0x00000000002003 (0x0000000000002000, Attributes(VALID | TABLE_OR_PAGE))
    0  -2  : 0
    3  -5  : 0x00000000003803 (0x0000000000003000, Attributes(VALID | TABLE_OR_PAGE | NON_GLOBAL))
    6      : 0x00000000006083 (0x0000000000006000, Attributes(VALID | TABLE_OR_PAGE | READ_ONLY))
    7      : 0
    8      : 0x00000000008083 (0x0000000000008000, Attributes(VALID | TABLE_OR_PAGE | READ_ONLY))
    9  -511: 0
  1  -511: 0
1  -511: 0
}"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn debug_roottable_contiguous_block() {
        let mut table = RootTable::<TargetAllocator>::new(
            TargetAllocator::new(0),
            1,
            TranslationRegime::El1And0,
            VaRange::Lower,
        );
        const BLOCK_SIZE: usize = PAGE_SIZE * 512;
        table
            .map_range(
                &MemoryRegion::new(BLOCK_SIZE * 3, BLOCK_SIZE * 6),
                PhysicalAddress(BLOCK_SIZE * 3),
                Attributes::VALID | Attributes::NON_GLOBAL,
                Constraints::empty(),
            )
            .unwrap();
        table
            .map_range(
                &MemoryRegion::new(BLOCK_SIZE * 6, BLOCK_SIZE * 7),
                PhysicalAddress(BLOCK_SIZE * 6),
                Attributes::VALID | Attributes::READ_ONLY,
                Constraints::empty(),
            )
            .unwrap();
        table
            .map_range(
                &MemoryRegion::new(BLOCK_SIZE * 8, BLOCK_SIZE * 9),
                PhysicalAddress(BLOCK_SIZE * 8),
                Attributes::VALID | Attributes::READ_ONLY,
                Constraints::empty(),
            )
            .unwrap();
        assert_eq!(
            format!("{table:?}"),
"RootTable { pa: 0x0000000000000000, translation_regime: El1And0, va_range: Lower, level: 1, table:
0      : 0x00000000001003 (0x0000000000001000, Attributes(VALID | TABLE_OR_PAGE))
  0  -2  : 0
  3  -5  : 0x00000000600801 (0x0000000000600000, Attributes(VALID | NON_GLOBAL))
  6      : 0x00000000c00081 (0x0000000000c00000, Attributes(VALID | READ_ONLY))
  7      : 0
  8      : 0x00000001000081 (0x0000000001000000, Attributes(VALID | READ_ONLY))
  9  -511: 0
1  -511: 0
}"
        );
    }
}
