// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Generic aarch64 page table manipulation functionality which doesn't assume anything about how
//! addresses are mapped.

pub mod attributes;

use crate::MapError;
#[cfg(feature = "alloc")]
use alloc::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use attributes::{Attributes, CommonAttributes};
use bitflags::bitflags;
use core::fmt::{self, Debug, Display, Formatter};
use core::marker::PhantomData;
use core::ops::{Add, Range, Sub};
use core::ptr::NonNull;
#[cfg(feature = "zerocopy")]
use zerocopy::{AsBytes, FromBytes, FromZeroes};

const PAGE_SHIFT: usize = 12;

/// The pagetable level at which all entries are page mappings.
const LEAF_LEVEL: usize = 3;

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
}

/// An aarch64 virtual address, the input type of a stage 1 page table.
#[cfg_attr(feature = "zerocopy", derive(AsBytes, FromBytes, FromZeroes))]
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct VirtualAddress(pub usize);

impl Display for VirtualAddress {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:#018x}", self.0)
    }
}

impl Debug for VirtualAddress {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "VirtualAddress({})", self)
    }
}

impl Sub for VirtualAddress {
    type Output = usize;

    fn sub(self, other: Self) -> Self::Output {
        self.0 - other.0
    }
}

impl Add<usize> for VirtualAddress {
    type Output = Self;

    fn add(self, other: usize) -> Self {
        Self(self.0 + other)
    }
}

impl Sub<usize> for VirtualAddress {
    type Output = Self;

    fn sub(self, other: usize) -> Self {
        Self(self.0 - other)
    }
}

/// A range of virtual addresses which may be mapped in a page table.
#[derive(Clone, Eq, PartialEq)]
pub struct MemoryRegion(Range<VirtualAddress>);

/// An aarch64 physical address or intermediate physical address, the output type of a stage 1 page
/// table.
#[cfg_attr(feature = "zerocopy", derive(AsBytes, FromBytes, FromZeroes))]
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct PhysicalAddress(pub usize);

impl Display for PhysicalAddress {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:#018x}", self.0)
    }
}

impl Debug for PhysicalAddress {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "PhysicalAddress({})", self)
    }
}

impl Sub for PhysicalAddress {
    type Output = usize;

    fn sub(self, other: Self) -> Self::Output {
        self.0 - other.0
    }
}

impl Add<usize> for PhysicalAddress {
    type Output = Self;

    fn add(self, other: usize) -> Self {
        Self(self.0 + other)
    }
}

impl Sub<usize> for PhysicalAddress {
    type Output = Self;

    fn sub(self, other: usize) -> Self {
        Self(self.0 - other)
    }
}

/// Returns the size in bytes of the address space covered by a single entry in the page table at
/// the given level.
pub(crate) fn granularity_at_level(level: usize) -> usize {
    PAGE_SIZE << ((LEAF_LEVEL - level) * BITS_PER_LEVEL)
}

/// An implementation of this trait needs to be provided to the mapping routines, so that the
/// physical addresses used in the page tables can be converted into virtual addresses that can be
/// used to access their contents from the code.
pub trait Translation<A: Attributes> {
    /// Allocates a zeroed page, which is already mapped, to be used for a new subtable of some
    /// pagetable. Returns both a pointer to the page and its physical address.
    fn allocate_table(&mut self) -> (NonNull<PageTable<A>>, PhysicalAddress);

    /// Deallocates the page which was previous allocated by [`allocate_table`](Self::allocate_table).
    ///
    /// # Safety
    ///
    /// The memory must have been allocated by `allocate_table` on the same `Translation`, and not
    /// yet deallocated.
    unsafe fn deallocate_table(&mut self, page_table: NonNull<PageTable<A>>);

    /// Given the physical address of a subtable, returns the virtual address at which it is mapped.
    fn physical_to_virtual(&self, pa: PhysicalAddress) -> NonNull<PageTable<A>>;
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

    fn split(&self, level: usize) -> ChunkedIterator {
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
pub struct RootTable<T: Translation<A>, A: Attributes> {
    table: PageTableWithLevel<T, A>,
    translation: T,
    pa: PhysicalAddress,
    va_range: VaRange,
}

impl<T: Translation<A>, A: Attributes> RootTable<T, A> {
    /// Creates a new page table starting at the given root level.
    ///
    /// The level must be between 0 and 3; level -1 (for 52-bit addresses with LPA2) is not
    /// currently supported by this library. The value of `TCR_EL1.T0SZ` must be set appropriately
    /// to match.
    pub fn new(mut translation: T, level: usize, va_range: VaRange) -> Self {
        if level > LEAF_LEVEL {
            panic!("Invalid root table level {}.", level);
        }
        if !A::TRANSLATION_REGIME.supports_asid() && va_range != VaRange::Lower {
            panic!(
                "{:?} doesn't have an upper virtual address range.",
                A::TRANSLATION_REGIME
            );
        }
        let (table, pa) = PageTableWithLevel::new(&mut translation, level);
        RootTable {
            table,
            translation,
            pa,
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
        flags: A,
        constraints: Constraints,
    ) -> Result<(), MapError<A>> {
        if flags.contains(CommonAttributes::TABLE_OR_PAGE.into()) {
            return Err(MapError::InvalidFlags(
                CommonAttributes::TABLE_OR_PAGE.into(),
            ));
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
    ///
    /// Returns [`MapError::BreakBeforeMakeViolation'] if the range intersects with live mappings,
    /// and modifying those would violate architectural break-before-make (BBM) requirements.
    pub fn modify_range<F>(&mut self, range: &MemoryRegion, f: &F) -> Result<(), MapError<A>>
    where
        F: Fn(&MemoryRegion, &mut Descriptor<A>, usize) -> Result<(), ()> + ?Sized,
    {
        self.verify_region(range)?;
        self.table.modify_range(&mut self.translation, range, f)
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
    pub fn walk_range<F>(&self, range: &MemoryRegion, f: &mut F) -> Result<(), MapError<A>>
    where
        F: FnMut(&MemoryRegion, &Descriptor<A>, usize) -> Result<(), ()>,
    {
        self.visit_range(range, &mut |mr, desc, level| {
            f(mr, desc, level).map_err(|_| MapError::PteUpdateFault(*desc))
        })
    }

    // Private version of `walk_range` using a closure that returns MapError on error
    pub(crate) fn visit_range<F>(&self, range: &MemoryRegion, f: &mut F) -> Result<(), MapError<A>>
    where
        F: FnMut(&MemoryRegion, &Descriptor<A>, usize) -> Result<(), MapError<A>>,
    {
        self.verify_region(range)?;
        self.table.visit_range(&self.translation, range, f)
    }

    /// Returns the level of mapping used for the given virtual address:
    /// - `None` if it is unmapped
    /// - `Some(LEAF_LEVEL)` if it is mapped as a single page
    /// - `Some(level)` if it is mapped as a block at `level`
    #[cfg(test)]
    pub(crate) fn mapping_level(&self, va: VirtualAddress) -> Option<usize> {
        self.table.mapping_level(&self.translation, va)
    }

    /// Checks whether the region is within range of the page table.
    fn verify_region(&self, region: &MemoryRegion) -> Result<(), MapError<A>> {
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

impl<T: Translation<A>, A: Attributes> Debug for RootTable<T, A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        writeln!(
            f,
            "RootTable {{ pa: {}, level: {}, table:",
            self.pa, self.table.level
        )?;
        self.table.fmt_indented(f, &self.translation, 0)?;
        write!(f, "}}")
    }
}

impl<T: Translation<A>, A: Attributes> Drop for RootTable<T, A> {
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
struct PageTableWithLevel<T: Translation<A>, A: Attributes> {
    table: NonNull<PageTable<A>>,
    level: usize,
    _translation: PhantomData<T>,
}

// SAFETY: The underlying PageTable is process-wide and can be safely accessed from any thread
// with appropriate synchronization. This type manages ownership for the raw pointer.
unsafe impl<T: Translation<A> + Send, A: Attributes + Send> Send for PageTableWithLevel<T, A> {}

// SAFETY: &Self only allows reading from the page table, which is safe to do from any thread.
unsafe impl<T: Translation<A> + Sync, A: Attributes + Sync> Sync for PageTableWithLevel<T, A> {}

impl<T: Translation<A>, A: Attributes> PageTableWithLevel<T, A> {
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

    fn from_pointer(table: NonNull<PageTable<A>>, level: usize) -> Self {
        Self {
            table,
            level,
            _translation: PhantomData,
        }
    }

    /// Returns a reference to the descriptor corresponding to a given virtual address.
    fn get_entry(&self, va: VirtualAddress) -> &Descriptor<A> {
        let shift = PAGE_SHIFT + (LEAF_LEVEL - self.level) * BITS_PER_LEVEL;
        let index = (va.0 >> shift) % (1 << BITS_PER_LEVEL);
        // SAFETY: We know that the pointer is properly aligned, dereferenced and initialised, and
        // nothing else can access the page table while we hold a mutable reference to the
        // PageTableWithLevel (assuming it is not currently active).
        let table = unsafe { self.table.as_ref() };
        &table.entries[index]
    }

    /// Returns a mutable reference to the descriptor corresponding to a given virtual address.
    fn get_entry_mut(&mut self, va: VirtualAddress) -> &mut Descriptor<A> {
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
        entry: &mut Descriptor<A>,
        level: usize,
    ) -> Self {
        let granularity = granularity_at_level(level);
        let old = *entry;
        let (mut subtable, subtable_pa) = Self::new(translation, level + 1);
        if let Some(old_flags) = old.flags() {
            let old_pa = old.output_address();
            if !old_flags.contains(CommonAttributes::TABLE_OR_PAGE.into())
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
        }
        entry.set(
            subtable_pa,
            CommonAttributes::TABLE_OR_PAGE | CommonAttributes::VALID,
        );
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
        flags: A,
        constraints: Constraints,
    ) {
        let level = self.level;
        let granularity = granularity_at_level(level);

        for chunk in range.split(level) {
            let entry = self.get_entry_mut(chunk.0.start);

            if level == LEAF_LEVEL {
                // Put down a page mapping.
                entry.set(pa, flags | A::from(CommonAttributes::TABLE_OR_PAGE));
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
            if table.entries[i].value == 0 {
                let first_zero = i;
                while i < table.entries.len() && table.entries[i].value == 0 {
                    i += 1;
                }
                if i - 1 == first_zero {
                    writeln!(f, "{:indentation$}{: <WIDTH$}: 0", "", first_zero)?;
                } else {
                    writeln!(f, "{:indentation$}{: <WIDTH$}-{}: 0", "", first_zero, i - 1)?;
                }
            } else {
                writeln!(
                    f,
                    "{:indentation$}{: <WIDTH$}: {:?}",
                    "", i, table.entries[i],
                )?;
                if let Some(subtable) = table.entries[i].subtable(translation, self.level) {
                    subtable.fmt_indented(f, translation, indentation + 2)?;
                }
                i += 1;
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
        for entry in table.entries {
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
        range: &MemoryRegion,
        f: &F,
    ) -> Result<(), MapError<A>>
    where
        F: Fn(&MemoryRegion, &mut Descriptor<A>, usize) -> Result<(), ()> + ?Sized,
    {
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
                subtable.modify_range(translation, &chunk, f)?;
            } else {
                f(&chunk, entry, level).map_err(|_| MapError::PteUpdateFault(*entry))?;
            }
        }
        Ok(())
    }

    /// Walks a range of page table entries and passes each one to a caller provided function.
    /// If the function returns an error, the walk is terminated and the error value is passed on
    fn visit_range<F, E>(&self, translation: &T, range: &MemoryRegion, f: &mut F) -> Result<(), E>
    where
        F: FnMut(&MemoryRegion, &Descriptor<A>, usize) -> Result<(), E>,
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
    #[cfg(test)]
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
#[cfg_attr(feature = "zerocopy", derive(FromZeroes))]
pub struct PageTable<A: Attributes> {
    entries: [Descriptor<A>; 1 << BITS_PER_LEVEL],
}

#[cfg(feature = "zerocopy")]
// SAFETY: PageTable is repr(C), has no padding, and we enforce that its field is AsBytes.
// Unfortunately the derive macro doesn't work here because of the generic parameter, but Descriptor
// only uses it in a PhantomData so it doesn't actually affect the representation.
unsafe impl<A: Attributes> AsBytes for PageTable<A>
where
    Descriptor<A>: AsBytes,
{
    fn only_derive_is_allowed_to_implement_this_trait() {}
}

impl<A: Attributes> PageTable<A> {
    /// Allocates a new zeroed, appropriately-aligned pagetable on the heap using the global
    /// allocator and returns a pointer to it.
    #[cfg(feature = "alloc")]
    pub fn new() -> NonNull<Self> {
        // SAFETY: Zeroed memory is a valid initialisation for a PageTable.
        unsafe { allocate_zeroed() }
    }
}

/// An entry in a page table.
///
/// A descriptor may be:
///   - Invalid, i.e. the virtual address range is unmapped
///   - A page mapping, if it is in the lowest level page table.
///   - A block mapping, if it is not in the lowest level page table.
///   - A pointer to a lower level pagetable, if it is not in the lowest level page table.
#[cfg_attr(feature = "zerocopy", derive(AsBytes, FromZeroes))]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Descriptor<A: Attributes> {
    value: usize,
    _attributes: PhantomData<A>,
}

impl<A: Attributes> Descriptor<A> {
    const PHYSICAL_ADDRESS_BITMASK: usize = !(PAGE_SIZE - 1) & !(0xffff << 48);

    #[cfg(test)]
    const fn new(value: usize) -> Self {
        Self {
            value,
            _attributes: PhantomData,
        }
    }

    pub(crate) fn output_address(self) -> PhysicalAddress {
        PhysicalAddress(self.value & Self::PHYSICAL_ADDRESS_BITMASK)
    }

    /// Returns the flags of this page table entry, or `None` if its state does not
    /// contain a valid set of flags.
    pub fn flags(self) -> Option<A> {
        A::from_bits(self.value & !Self::PHYSICAL_ADDRESS_BITMASK)
    }

    /// Modifies the page table entry by setting or clearing its flags.
    /// Panics when attempting to convert a table descriptor into a block/page descriptor or vice
    /// versa - this is not supported via this API.
    pub fn modify_flags(&mut self, set: A, clear: A) {
        let flags = (self.value | set.bits()) & !clear.bits();

        if (self.value ^ flags) & CommonAttributes::TABLE_OR_PAGE.bits() != 0 {
            panic!("Cannot convert between table and block/page descriptors\n");
        }
        self.value = flags;
    }

    /// Returns `true` if [`Attributes::VALID`] is set on this entry, e.g. if the entry is mapped.
    pub fn is_valid(self) -> bool {
        (self.value & CommonAttributes::VALID.bits()) != 0
    }

    /// Returns `true` if this is a valid entry pointing to a next level translation table or a page.
    pub fn is_table_or_page(self) -> bool {
        if let Some(flags) = self.flags() {
            flags.contains((CommonAttributes::TABLE_OR_PAGE | CommonAttributes::VALID).into())
        } else {
            false
        }
    }

    pub(crate) fn set(&mut self, pa: PhysicalAddress, flags: impl Into<A>) {
        self.value = (pa.0 & Self::PHYSICAL_ADDRESS_BITMASK) | flags.into().bits();
    }

    fn subtable<T: Translation<A>>(
        self,
        translation: &T,
        level: usize,
    ) -> Option<PageTableWithLevel<T, A>> {
        if level < LEAF_LEVEL && self.is_table_or_page() {
            let output_address = self.output_address();
            let table = translation.physical_to_virtual(output_address);
            return Some(PageTableWithLevel::from_pointer(table, level + 1));
        }
        None
    }
}

impl<A: Attributes> Debug for Descriptor<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:#016x}", self.value)?;
        if self.is_valid() {
            if let Some(flags) = self.flags() {
                write!(f, " ({}, {:?})", self.output_address(), flags)?;
            }
        }
        Ok(())
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
    use super::attributes::AttributesEl1;
    #[cfg(feature = "alloc")]
    use super::attributes::{AttributesEl2, AttributesEl3};
    use super::*;
    #[cfg(feature = "alloc")]
    use crate::idmap::IdTranslation;
    #[cfg(feature = "alloc")]
    use alloc::{format, string::ToString, vec, vec::Vec};

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
        let desc = Descriptor::<AttributesEl1>::new(0usize);
        assert!(!desc.is_valid());
        assert!(!desc.flags().unwrap().contains(AttributesEl1::VALID));
    }

    #[test]
    fn set_descriptor() {
        const PHYSICAL_ADDRESS: usize = 0x12340000;
        let mut desc = Descriptor::<AttributesEl1>::new(0usize);
        assert!(!desc.is_valid());
        desc.set(
            PhysicalAddress(PHYSICAL_ADDRESS),
            AttributesEl1::TABLE_OR_PAGE
                | AttributesEl1::USER
                | AttributesEl1::SWFLAG_1
                | AttributesEl1::VALID,
        );
        assert!(desc.is_valid());
        assert_eq!(
            desc.flags().unwrap(),
            AttributesEl1::TABLE_OR_PAGE
                | AttributesEl1::USER
                | AttributesEl1::SWFLAG_1
                | AttributesEl1::VALID
        );
        assert_eq!(desc.output_address(), PhysicalAddress(PHYSICAL_ADDRESS));
    }

    #[test]
    fn modify_descriptor_flags() {
        let mut desc = Descriptor::<AttributesEl1>::new(0usize);
        assert!(!desc.is_valid());
        desc.set(
            PhysicalAddress(0x12340000),
            AttributesEl1::TABLE_OR_PAGE | AttributesEl1::USER | AttributesEl1::SWFLAG_1,
        );
        desc.modify_flags(
            AttributesEl1::DBM | AttributesEl1::SWFLAG_3,
            AttributesEl1::VALID | AttributesEl1::SWFLAG_1,
        );
        assert!(!desc.is_valid());
        assert_eq!(
            desc.flags().unwrap(),
            AttributesEl1::TABLE_OR_PAGE
                | AttributesEl1::USER
                | AttributesEl1::SWFLAG_3
                | AttributesEl1::DBM
        );
    }

    #[test]
    #[should_panic]
    fn modify_descriptor_table_or_page_flag() {
        let mut desc = Descriptor::<AttributesEl1>::new(0usize);
        assert!(!desc.is_valid());
        desc.set(
            PhysicalAddress(0x12340000),
            AttributesEl1::TABLE_OR_PAGE | AttributesEl1::USER | AttributesEl1::SWFLAG_1,
        );
        desc.modify_flags(AttributesEl1::VALID, AttributesEl1::TABLE_OR_PAGE);
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
        RootTable::<IdTranslation, AttributesEl2>::new(IdTranslation, 1, VaRange::Upper);
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[should_panic]
    fn no_el3_ttbr1() {
        RootTable::<IdTranslation, AttributesEl3>::new(IdTranslation, 1, VaRange::Upper);
    }
}
