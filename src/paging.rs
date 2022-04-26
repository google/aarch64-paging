// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Generic aarch64 page table manipulation functionality which doesn't assume anything about how
//! addresses are mapped.

use alloc::boxed::Box;
use core::alloc::Layout;
use core::fmt::{self, Debug, Display, Formatter};
use core::marker::PhantomData;
use core::ops::Range;

pub const PAGE_SHIFT: usize = 12;

/// The page size in bytes assumed by this library, 4 KiB.
pub const PAGE_SIZE: usize = 1 << PAGE_SHIFT;

pub const BITS_PER_LEVEL: usize = PAGE_SHIFT - 3;

/// An aarch64 virtual address, the input type of a stage 1 page table.
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct VirtualAddress(pub usize);

impl Display for VirtualAddress {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:#016x}", self.0)
    }
}

/// A range of virtual addresses which may be mapped in a page table.
#[derive(Clone, Eq, PartialEq)]
pub struct MemoryRegion(Range<VirtualAddress>);

/// An aarch64 physical address or intermediate physical address, the output type of a stage 1 page
/// table.
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct PhysicalAddress(pub usize);

impl Display for PhysicalAddress {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:#016x}", self.0)
    }
}

/// An implementation of this trait needs to be provided to the mapping routines, so that the
/// physical addresses used in the page tables can be converted into virtual addresses that can be
/// used to access their contents from the code.
pub trait Translation {
    fn virtual_to_physical(va: VirtualAddress) -> PhysicalAddress;
    fn physical_to_virtual(pa: PhysicalAddress) -> VirtualAddress;
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

    pub const fn start(&self) -> VirtualAddress {
        self.0.start
    }

    pub const fn end(&self) -> VirtualAddress {
        self.0.end
    }

    pub const fn len(&self) -> usize {
        self.0.end.0 - self.0.start.0
    }
}

fn get_zeroed_page() -> VirtualAddress {
    let layout = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).unwrap();
    let page = unsafe { alloc::alloc::alloc_zeroed(layout) };
    if page.is_null() {
        panic!("Out of memory!");
    }
    VirtualAddress(page as usize)
}

/// A complete hierarchy of page tables including all levels.
#[derive(Debug)]
pub struct RootTable<T: Translation> {
    table: Box<PageTable<T>>,
    level: usize,
}

impl<T: Translation> RootTable<T> {
    /// Creates a new page table starting at the given root level.
    pub fn new(level: usize) -> Self {
        RootTable {
            table: unsafe {
                // We need to use from_raw() here to avoid allocating
                // on the stack and copying into the box
                Box::<PageTable<T>>::from_raw(get_zeroed_page().0 as *mut _)
            },
            level,
        }
    }

    /// Recursively maps a range into the pagetable hierarchy starting at the root level.
    pub fn map_range(&mut self, range: &MemoryRegion, flags: Attributes) {
        self.table.map_range(range, flags, self.level);
    }

    /// Returns the physical address of the root table in memory.
    pub fn to_physical(&self) -> PhysicalAddress {
        self.table.to_physical()
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

impl MemoryRegion {
    fn split(&self, level: usize) -> ChunkedIterator {
        ChunkedIterator {
            range: self,
            granularity: PAGE_SIZE << ((3 - level) * BITS_PER_LEVEL),
            start: self.0.start.0,
        }
    }

    /// Returns whether this region can be mapped at 'level' using block mappings only.
    fn is_block(&self, level: usize) -> bool {
        let gran = PAGE_SIZE << ((3 - level) * BITS_PER_LEVEL);
        (self.0.start.0 | self.0.end.0) & (gran - 1) == 0
    }
}

bitflags! {
    /// Attribute bits for a mapping in a page table.
    pub struct Attributes: usize {
        const VALID         = 1 << 0;
        const TABLE_OR_PAGE = 1 << 1;

        // The following memory types assume that the MAIR registers
        // have been programmed accordingly.
        const DEVICE_NGNRE  = 0 << 2;
        const NORMAL        = 1 << 2 | 3 << 8; // inner shareable

        const USER          = 1 << 6;
        const READ_ONLY     = 1 << 7;
        const ACCESSED      = 1 << 10;
        const NON_GLOBAL    = 1 << 11;
        const EXECUTE_NEVER = 3 << 53;
    }
}

/// A single level of a page table.
#[repr(C, align(4096))]
pub struct PageTable<T> {
    entries: [Descriptor; 1 << BITS_PER_LEVEL],
    _phantom_data: PhantomData<T>,
}

/// An entry in a page table.
///
/// A descriptor may be:
///   - Invalid, i.e. the virtual address range is unmapped
///   - A page mapping, if it is in the lowest level page table.
///   - A block mapping, if it is not in the lowest level page table.
///   - A pointer to a lower level pagetable, if it is not in the lowest level page table.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Descriptor(usize);

impl Descriptor {
    fn output_address(&self) -> Option<PhysicalAddress> {
        if self.is_valid() {
            Some(PhysicalAddress(
                self.0 & (!(PAGE_SIZE - 1) & !(0xffff << 48)),
            ))
        } else {
            None
        }
    }

    fn flags(self) -> Option<Attributes> {
        if self.is_valid() {
            Attributes::from_bits(self.0 & ((PAGE_SIZE - 1) | (0xffff << 48)))
        } else {
            None
        }
    }

    fn is_valid(self) -> bool {
        (self.0 & Attributes::VALID.bits()) != 0
    }

    fn is_table(self) -> bool {
        if let Some(flags) = self.flags() {
            flags.contains(Attributes::TABLE_OR_PAGE)
        } else {
            false
        }
    }

    fn set(&mut self, pa: PhysicalAddress, flags: Attributes) {
        self.0 = pa.0 | (flags | Attributes::VALID).bits();
    }

    fn subtable<T: Translation>(&self) -> Option<&mut PageTable<T>> {
        if self.is_table() {
            if let Some(output_address) = self.output_address() {
                let va = T::physical_to_virtual(output_address);
                return Some(unsafe { &mut *(va.0 as *mut PageTable<T>) });
            }
        }
        None
    }
}

impl Debug for Descriptor {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:#016x}", self.0)?;
        if let (Some(flags), Some(address)) = (self.flags(), self.output_address()) {
            write!(f, " ({}, {:?})", address, flags)?;
        }
        Ok(())
    }
}

impl<T: Translation> Debug for PageTable<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        writeln!(f)?;
        self.fmt_indented(f, 0)
    }
}

impl<T: Translation> PageTable<T> {
    /// Returns the physical address of this page table in memory.
    pub fn to_physical(&self) -> PhysicalAddress {
        T::virtual_to_physical(VirtualAddress(self as *const _ as usize))
    }

    fn get_entry_mut(&mut self, va: usize, level: usize) -> &mut Descriptor {
        let shift = PAGE_SHIFT + (3 - level) * BITS_PER_LEVEL;
        let index = (va >> shift) % (1 << BITS_PER_LEVEL);
        &mut self.entries[index]
    }

    fn map_range(&mut self, range: &MemoryRegion, flags: Attributes, level: usize) {
        assert!(level <= 3);
        let mut pa = T::virtual_to_physical(range.start());

        for chunk in range.split(level) {
            let entry = self.get_entry_mut(chunk.0.start.0, level);

            if level == 3 {
                // Put down a page mapping.
                entry.set(pa, flags | Attributes::ACCESSED | Attributes::TABLE_OR_PAGE);
            } else if chunk.is_block(level) && !entry.is_table() {
                // Rather than leak the entire subhierarchy, only put down
                // a block mapping if the region is not already covered by
                // a table mapping
                entry.set(pa, flags | Attributes::ACCESSED);
            } else {
                if !entry.is_table() {
                    let old = *entry;
                    let page = T::virtual_to_physical(get_zeroed_page());
                    entry.set(page, Attributes::TABLE_OR_PAGE);
                    if let Some(old_flags) = old.flags() {
                        let granularity = PAGE_SIZE << ((3 - level) * BITS_PER_LEVEL);
                        // Old was a valid block entry, so we need to split it
                        // Recreate the entire block in the newly added table
                        let a = align_down(chunk.0.start.0, granularity);
                        let b = align_up(chunk.0.end.0, granularity);
                        // We just made it into a table so subtable can't return None.
                        entry.subtable::<T>().unwrap().map_range(
                            &MemoryRegion::new(a, b),
                            old_flags,
                            level + 1,
                        );
                    }
                }
                // Either the entry was a table or it is now, so subtable can't return None.
                entry
                    .subtable::<T>()
                    .unwrap()
                    .map_range(&chunk, flags, level + 1);
            }
            pa.0 += chunk.len();
        }
    }

    fn fmt_indented(&self, f: &mut Formatter, indentation: usize) -> Result<(), fmt::Error> {
        let mut i = 0;
        while i < self.entries.len() {
            if self.entries[i].0 == 0 {
                let first_zero = i;
                while i < self.entries.len() && self.entries[i].0 == 0 {
                    i += 1;
                }
                if i - 1 == first_zero {
                    writeln!(f, "{:indentation$}{}: 0", "", first_zero)?;
                } else {
                    writeln!(f, "{:indentation$}{}-{}: 0", "", first_zero, i - 1)?;
                }
            } else {
                writeln!(f, "{:indentation$}{}: {:?}", "", i, self.entries[i])?;
                if let Some(subtable) = self.entries[i].subtable::<T>() {
                    subtable.fmt_indented(f, indentation + 2)?;
                }
                i += 1;
            }
        }
        Ok(())
    }
}

const fn align_down(value: usize, alignment: usize) -> usize {
    value & !(alignment - 1)
}

const fn align_up(value: usize, alignment: usize) -> usize {
    ((value - 1) | (alignment - 1)) + 1
}
