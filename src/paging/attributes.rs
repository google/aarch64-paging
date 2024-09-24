// Copyright 2024 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

use super::TranslationRegime;
use bitflags::{bitflags, Flags};
use core::{
    fmt::Debug,
    ops::{BitOr, BitXor, Sub},
};

/// Attribute bits for a mapping in a page table.
pub trait Attributes:
    Copy
    + Clone
    + Debug
    + Flags<Bits = usize>
    + Sub<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + From<CommonAttributes>
{
    const TRANSLATION_REGIME: TranslationRegime;
}

bitflags! {
    #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct CommonAttributes: usize{
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

        const NON_GLOBAL = 1 << 11;
    }
}

impl CommonAttributes {
    /// Mask for the bits determining the shareability of the mapping.
    pub const SHAREABILITY_MASK: Self = Self::INNER_SHAREABLE;

    /// Mask for the bits determining the attribute index of the mapping.
    pub const ATTRIBUTE_INDEX_MASK: Self = Self::ATTRIBUTE_INDEX_7;
}

bitflags! {
    /// Attribute bits for a mapping in a page table for the EL3 translation regime.
    #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct AttributesEl3: usize {
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
        const USER_RES1     = 1 << 6;
        const READ_ONLY     = 1 << 7;
        const ACCESSED      = 1 << 10;
        const NSE           = 1 << 11;
        const DBM           = 1 << 51;
        /// Execute-never.
        const XN            = 1 << 54;

        // Software flags in block and page descriptor entries.
        const SWFLAG_0 = 1 << 55;
        const SWFLAG_1 = 1 << 56;
        const SWFLAG_2 = 1 << 57;
        const SWFLAG_3 = 1 << 58;
    }
}

impl AttributesEl3 {
    pub const RES1: Self = Self::USER_RES1;
}

impl Attributes for AttributesEl3 {
    const TRANSLATION_REGIME: TranslationRegime = TranslationRegime::El3;
}

impl From<CommonAttributes> for AttributesEl3 {
    fn from(common: CommonAttributes) -> Self {
        Self::from_bits_retain(common.bits())
    }
}

bitflags! {
    /// Attribute bits for a mapping in a page table for the non-secure EL2 translation regime.
    #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct AttributesEl2: usize {
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
    }
}

impl Attributes for AttributesEl2 {
    const TRANSLATION_REGIME: TranslationRegime = TranslationRegime::El2;
}

impl From<CommonAttributes> for AttributesEl2 {
    fn from(common: CommonAttributes) -> Self {
        Self::from_bits_retain(common.bits())
    }
}

bitflags! {
    /// Attribute bits for a mapping in a page table for the EL1&0 translation regime.
    #[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct AttributesEl1: usize {
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
    }
}

impl Attributes for AttributesEl1 {
    const TRANSLATION_REGIME: TranslationRegime = TranslationRegime::El1And0;
}

impl From<CommonAttributes> for AttributesEl1 {
    fn from(common: CommonAttributes) -> Self {
        Self::from_bits_retain(common.bits())
    }
}
