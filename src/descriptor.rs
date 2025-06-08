// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Abstractions for page table descriptor and the physical addresses and attributes they may
//! describe

use crate::Translation;
use crate::paging::LEAF_LEVEL;
use crate::paging::PAGE_SIZE;
use crate::paging::PageTableWithLevel;

use bitflags::bitflags;
use core::fmt::{self, Debug, Display, Formatter};
use core::ops::{Add, Sub};
use core::sync::atomic::{AtomicUsize, Ordering};

/// An aarch64 virtual address, the input type of a stage 1 page table.
#[derive(Copy, Clone, Default, Eq, Ord, PartialEq, PartialOrd)]
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

/// An aarch64 physical address or intermediate physical address, the output type of a stage 1 page
/// table.
#[derive(Copy, Clone, Default, Eq, Ord, PartialEq, PartialOrd)]
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

bitflags! {
    /// Attribute bits for a mapping in a page table.
    #[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct Attributes: usize {
        const VALID         = 1 << 0;
        const TABLE_OR_PAGE = 1 << 1;

        const ATTRIBUTE_INDEX_0 = 0 << 2;
        const ATTRIBUTE_INDEX_1 = 1 << 2;
        const ATTRIBUTE_INDEX_2 = 2 << 2;
        const ATTRIBUTE_INDEX_3 = 3 << 2;
        const ATTRIBUTE_INDEX_4 = 4 << 2;
        const ATTRIBUTE_INDEX_5 = 5 << 2;
        const ATTRIBUTE_INDEX_6 = 6 << 2;
        const ATTRIBUTE_INDEX_7 = 7 << 2;

        const OUTER_SHAREABLE = 2 << 8;
        const INNER_SHAREABLE = 3 << 8;

        const NS            = 1 << 5;
        const USER          = 1 << 6;
        const READ_ONLY     = 1 << 7;
        const ACCESSED      = 1 << 10;
        const NON_GLOBAL    = 1 << 11;
        /// Guarded Page - indirect forward edge jumps expect an appropriate BTI landing pad.
        const GP            = 1 << 50;
        const DBM           = 1 << 51;
        /// Privileged Execute-never, if two privilege levels are supported.
        const PXN           = 1 << 53;
        /// Unprivileged Execute-never, or just Execute-never if only one privilege level is
        /// supported.
        const UXN           = 1 << 54;

        // Software flags in block and page descriptor entries.
        const SWFLAG_0 = 1 << 55;
        const SWFLAG_1 = 1 << 56;
        const SWFLAG_2 = 1 << 57;
        const SWFLAG_3 = 1 << 58;

        const PXN_TABLE = 1 << 59;
        const XN_TABLE = 1 << 60;
        const AP_TABLE_NO_EL0 = 1 << 61;
        const AP_TABLE_NO_WRITE = 1 << 62;
        const NS_TABLE = 1 << 63;
    }
}

impl Attributes {
    /// Mask for the bits determining the shareability of the mapping.
    pub const SHAREABILITY_MASK: Self = Self::INNER_SHAREABLE;

    /// Mask for the bits determining the attribute index of the mapping.
    pub const ATTRIBUTE_INDEX_MASK: Self = Self::ATTRIBUTE_INDEX_7;
}

pub(crate) type DescriptorBits = usize;

/// An entry in a page table.
///
/// A descriptor may be:
///   - Invalid, i.e. the virtual address range is unmapped
///   - A page mapping, if it is in the lowest level page table.
///   - A block mapping, if it is not in the lowest level page table.
///   - A pointer to a lower level pagetable, if it is not in the lowest level page table.
#[repr(C)]
pub struct Descriptor(pub(crate) AtomicUsize);

impl Descriptor {
    /// An empty (i.e. 0) descriptor.
    pub const EMPTY: Self = Self(AtomicUsize::new(0));

    const PHYSICAL_ADDRESS_BITMASK: usize = !(PAGE_SIZE - 1) & !(0xffff << 48);

    /// Returns the contents of a descriptor which may be potentially live
    /// Use acquire semantics so that the load is not reordered with subsequent loads
    pub(crate) fn bits(&self) -> DescriptorBits {
        self.0.load(Ordering::Acquire)
    }

    /// Returns the physical address that this descriptor refers to if it is valid.
    ///
    /// Depending on the flags this could be the address of a subtable, a mapping, or (if it is not
    /// a valid mapping) entirely arbitrary.
    pub fn output_address(&self) -> PhysicalAddress {
        PhysicalAddress(self.bits() & Self::PHYSICAL_ADDRESS_BITMASK)
    }

    /// Returns the flags of this page table entry, or `None` if its state does not
    /// contain a valid set of flags.
    pub fn flags(&self) -> Attributes {
        Attributes::from_bits_retain(self.bits() & !Self::PHYSICAL_ADDRESS_BITMASK)
    }

    /// Modifies the page table entry by setting or clearing its flags.
    /// Panics when attempting to convert a table descriptor into a block/page descriptor or vice
    /// versa - this is not supported via this API.
    pub fn modify_flags(&mut self, set: Attributes, clear: Attributes) {
        let oldval = self.bits();
        let flags = (oldval | set.bits()) & !clear.bits();

        if (oldval ^ flags) & Attributes::TABLE_OR_PAGE.bits() != 0 {
            panic!("Cannot convert between table and block/page descriptors\n");
        }

        self.0.store(flags, Ordering::Release);
    }

    /// Returns `true` if [`Attributes::VALID`] is set on this entry, e.g. if the entry is mapped.
    pub fn is_valid(&self) -> bool {
        (self.bits() & Attributes::VALID.bits()) != 0
    }

    /// Returns `true` if this is a valid entry pointing to a next level translation table or a page.
    pub fn is_table_or_page(&self) -> bool {
        self.flags()
            .contains(Attributes::TABLE_OR_PAGE | Attributes::VALID)
    }

    pub(crate) fn set(&mut self, pa: PhysicalAddress, flags: Attributes) {
        self.0.store(
            (pa.0 & Self::PHYSICAL_ADDRESS_BITMASK) | flags.bits(),
            Ordering::Release,
        );
    }

    pub(crate) fn subtable<T: Translation>(
        &self,
        translation: &T,
        level: usize,
    ) -> Option<PageTableWithLevel<T>> {
        if level < LEAF_LEVEL && self.is_table_or_page() {
            let output_address = self.output_address();
            let table = translation.physical_to_virtual(output_address);
            return Some(PageTableWithLevel::from_pointer(table, level + 1));
        }
        None
    }

    pub(crate) fn clone(&self) -> Self {
        Descriptor(AtomicUsize::new(self.bits()))
    }
}

impl Debug for Descriptor {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "{:#016x}", self.bits())?;
        if self.is_valid() {
            write!(f, " ({}, {:?})", self.output_address(), self.flags())?;
        }
        Ok(())
    }
}
