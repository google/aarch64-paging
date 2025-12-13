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
        Self::output_address_from_bits(self.bits())
    }

    fn output_address_from_bits(bits: DescriptorBits) -> PhysicalAddress {
        PhysicalAddress(bits & Self::PHYSICAL_ADDRESS_BITMASK)
    }

    fn flags_from_bits(bits: DescriptorBits) -> Attributes {
        Attributes::from_bits_retain(bits & !Self::PHYSICAL_ADDRESS_BITMASK)
    }

    /// Returns the flags of this page table entry, or `None` if its state does not
    /// contain a valid set of flags.
    pub fn flags(&self) -> Attributes {
        Self::flags_from_bits(self.bits())
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

enum DescriptorEnum<'a> {
    /// A descriptor that is part of a set of page tables that are currently in use by one of the
    /// CPUs
    Active(&'a mut Descriptor),

    /// A descriptor that is part of a set of page tables that are currently inactive. This means
    /// TLB maintenance may be elided until the next time the page tables are made active.
    Inactive(&'a mut Descriptor),

    /// A descriptor that does not actually represent an entry in a page table. It permits updaters
    /// taking an UpdatableDescriptor to be called for a dry run to observe their effect without
    /// the need to pass an actual descriptor.
    ActiveClone(DescriptorBits),
}

pub struct UpdatableDescriptor<'a> {
    descriptor: DescriptorEnum<'a>,
    level: usize,
    updated: bool,
}

impl<'a> UpdatableDescriptor<'a> {
    /// Creates a new wrapper around a real descriptor that may or may not be live
    pub(crate) fn new(desc: &'a mut Descriptor, level: usize, live: bool) -> Self {
        Self {
            descriptor: if live {
                DescriptorEnum::Active(desc)
            } else {
                DescriptorEnum::Inactive(desc)
            },
            level: level,
            updated: false,
        }
    }

    /// Creates a new wrapper around an ActiveClone descriptor, which is used to observe the
    /// effect of user provided updater functions without applying them to actual descriptors
    pub(crate) fn clone_from(d: &Descriptor, level: usize) -> Self {
        Self {
            descriptor: DescriptorEnum::ActiveClone(d.bits()),
            level: level,
            updated: false,
        }
    }

    /// Returns the level in the page table hierarchy at which this descriptor appears
    pub fn level(&self) -> usize {
        self.level
    }

    /// Whether the descriptor was updated as a result of a call to set() or modify_flags()
    pub fn updated(&self) -> bool {
        self.updated
    }

    /// Returns whether this descriptor represents a table mapping. In this case, the output address
    /// refers to a next level table.
    pub fn is_table(&self) -> bool {
        self.level < 3 && self.flags().contains(Attributes::TABLE_OR_PAGE)
    }

    /// Returns the bit representation of the underlying descriptor
    pub fn bits(&self) -> DescriptorBits {
        match &self.descriptor {
            DescriptorEnum::Active(d) | DescriptorEnum::Inactive(d) => d.bits(),
            DescriptorEnum::ActiveClone(d) => *d,
        }
    }

    /// Assigns the underlying descriptor according to `pa` and `flags, provided that doing so is
    /// permitted under BBM rules
    pub fn set(&mut self, pa: PhysicalAddress, flags: Attributes) -> Result<(), ()> {
        if !self.bbm_permits_update(pa, flags) {
            return Err(());
        }
        let val = (pa.0 & Descriptor::PHYSICAL_ADDRESS_BITMASK) | flags.bits();
        match &mut self.descriptor {
            DescriptorEnum::Active(d) | DescriptorEnum::Inactive(d) => {
                self.updated |= val != d.0.swap(val, Ordering::Release)
            }
            DescriptorEnum::ActiveClone(d) => {
                self.updated |= *d != val;
                *d = val
            }
        };
        Ok(())
    }

    /// Returns the physical address to which this descriptor refers
    ///
    /// Depending on the flags this could be the address of a subtable, a mapping, or (if it is not
    /// a valid mapping) entirely arbitrary.
    pub fn output_address(&self) -> PhysicalAddress {
        Descriptor::output_address_from_bits(self.bits())
    }

    /// Returns the flags of this descriptor
    pub fn flags(&self) -> Attributes {
        Descriptor::flags_from_bits(self.bits())
    }

    /// Returns whether this descriptor should be considered live and valid, in which case BBM
    /// rules need to be checked. ActiveClone() variants are explicitly intended for checking BBM
    /// rules, so they are considered live by this API
    fn is_live_and_valid(&self) -> bool {
        match &self.descriptor {
            DescriptorEnum::Inactive(_) => false,
            _ => self.flags().contains(Attributes::VALID),
        }
    }

    /// Returns whether BBM permits setting the flags on this descriptor to `flags`
    fn bbm_permits_update(&self, pa: PhysicalAddress, flags: Attributes) -> bool {
        if !self.is_live_and_valid() || !flags.contains(Attributes::VALID) {
            return true;
        }

        // Modifying the output address on a live valid descriptor is not allowed
        if pa != self.output_address() {
            return false;
        }

        // Masks of bits that may be set resp. cleared on a live, valid mapping without BBM
        let clear_allowed_mask = Attributes::VALID
            | Attributes::READ_ONLY
            | Attributes::ACCESSED
            | Attributes::DBM
            | Attributes::PXN
            | Attributes::UXN
            | Attributes::SWFLAG_0
            | Attributes::SWFLAG_1
            | Attributes::SWFLAG_2
            | Attributes::SWFLAG_3;
        let set_allowed_mask = clear_allowed_mask | Attributes::NON_GLOBAL;

        (!self.flags() & flags & !set_allowed_mask).is_empty()
            && (self.flags() & !flags & !clear_allowed_mask).is_empty()
    }

    /// Modifies the descriptor by setting or clearing its flags.
    pub fn modify_flags(&mut self, set: Attributes, clear: Attributes) -> Result<(), ()> {
        let oldval = self.flags();
        let flags = (oldval | set) & !clear;

        if (oldval ^ flags).contains(Attributes::TABLE_OR_PAGE) {
            // Cannot convert between table and block/page descriptors, regardless of whether or
            // not BBM permits this and whether the entry is live, given that doing so would
            // corrupt our data strucutures.
            return Err(());
        }
        let oa = self.output_address();
        self.set(oa, flags)
    }
}
