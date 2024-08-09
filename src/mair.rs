// Copyright 2024 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Types for Memory Attribute Indirection Register values.
//!
//! The main use for these is building a constant MAIR value in a readable structured way.
//!
//! # Example
//!
//! ```
//! use aarch64_paging::mair::{Mair, MairAttribute, NormalMemory};
//!
//! const MAIR: Mair = Mair::EMPTY
//!     .with_attribute(0, MairAttribute::DEVICE_NGNRE)
//!     .with_attribute(
//!         1,
//!         MairAttribute::normal(NormalMemory::NonCacheable, NormalMemory::NonCacheable),
//!     )
//!     .with_attribute(
//!         2,
//!         MairAttribute::normal(
//!             NormalMemory::WriteBackNonTransientReadWriteAllocate,
//!             NormalMemory::WriteBackNonTransientReadWriteAllocate,
//!         ),
//!     );
//! ```

use core::fmt::{self, Display, Formatter, Write};

/// A Memory Attribute Indirection Register value.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Mair(pub u64);

impl Mair {
    /// A 0 MAIR value.
    pub const EMPTY: Self = Self(0);

    /// Constructs a new MAIR value from a set of attributes.
    pub const fn new(attributes: [MairAttribute; 8]) -> Self {
        Self(
            attributes[0].0 as u64
                | (attributes[1].0 as u64) << 8
                | (attributes[2].0 as u64) << 16
                | (attributes[3].0 as u64) << 24
                | (attributes[4].0 as u64) << 32
                | (attributes[5].0 as u64) << 40
                | (attributes[6].0 as u64) << 48
                | (attributes[7].0 as u64) << 56,
        )
    }

    /// Sets the attribute at the given index, returning the new MAIR value.
    pub const fn with_attribute(self, index: u8, attribute: MairAttribute) -> Self {
        assert!(index < 8);
        let offset = index * 8;
        Self(self.0 & !(0xff << offset) | (attribute.0 as u64) << offset)
    }

    /// Breaks a MAIR value down into its individual attributes.
    pub const fn attributes(self) -> [MairAttribute; 8] {
        [
            MairAttribute(self.0 as u8),
            MairAttribute((self.0 >> 8) as u8),
            MairAttribute((self.0 >> 16) as u8),
            MairAttribute((self.0 >> 24) as u8),
            MairAttribute((self.0 >> 32) as u8),
            MairAttribute((self.0 >> 40) as u8),
            MairAttribute((self.0 >> 48) as u8),
            MairAttribute((self.0 >> 56) as u8),
        ]
    }
}

impl From<Mair> for u64 {
    fn from(value: Mair) -> Self {
        value.0
    }
}

impl From<[MairAttribute; 8]> for Mair {
    fn from(attributes: [MairAttribute; 8]) -> Self {
        Self::new(attributes)
    }
}

impl From<Mair> for [MairAttribute; 8] {
    fn from(value: Mair) -> Self {
        value.attributes()
    }
}

impl Display for Mair {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_char('{')?;
        for (i, attribute) in self.attributes().into_iter().enumerate() {
            if i != 0 {
                f.write_str("; ")?;
            }
            write!(f, "{}: {}", i, attribute)?;
        }
        f.write_char('}')?;
        Ok(())
    }
}

/// A single field in the MAIR.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MairAttribute(pub u8);

impl MairAttribute {
    /// Device-nGnRnE memory.
    pub const DEVICE_NGNRNE: Self = Self(0b0000_0000);
    /// Device-nGnRE  memory.
    pub const DEVICE_NGNRE: Self = Self(0b0000_0100);
    /// Device-nGRE memory.
    pub const DEVICE_NGRE: Self = Self(0b0000_1000);
    /// Device-GRE memory.
    pub const DEVICE_GRE: Self = Self(0b0000_1100);

    /// Returns a MAIR attribute for normal memory.
    pub const fn normal(inner: NormalMemory, outer: NormalMemory) -> Self {
        Self((outer as u8) << 4 | inner as u8)
    }
}

impl Display for MairAttribute {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            Self::DEVICE_NGNRNE => f.write_str("Device-nGnRnE"),
            Self::DEVICE_NGNRE => f.write_str("Device-nGnRE"),
            Self::DEVICE_NGRE => f.write_str("Device-nGRE"),
            Self::DEVICE_GRE => f.write_str("Device-GRE"),
            Self(value) => {
                let inner = value & 0x0f;
                let outer = value >> 4;
                if let (Ok(inner), Ok(outer)) =
                    (NormalMemory::try_from(inner), NormalMemory::try_from(outer))
                {
                    write!(f, "Normal, Inner {}, Outer {}", inner, outer)
                } else {
                    write!(f, "Unpredictable ({:#04x})", value)
                }
            }
        }
    }
}

/// The inner or outer attributes of normal memory.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum NormalMemory {
    /// Write-through transient, write-allocate
    WriteThroughTransientWriteAllocate = 0b0001,
    /// Write-through transient, read-allocate
    WriteThroughTransientReadAllocate = 0b0010,
    /// Write-through transient, read-write-allocate
    WriteThroughTransientReadWriteAllocate = 0b0011,
    /// Non-Cacheable
    NonCacheable = 0b0100,
    /// Write-back transient, write-allocate
    WriteBackTransientWriteAllocate = 0b0101,
    /// Write-back transient, read-allocate
    WriteBackTransientReadAllocate = 0b0110,
    /// Write-back transient, read-write-allocate
    WriteBackTransientReadWriteAllocate = 0b0111,
    /// Write-through non-transient, do not allocate
    WriteThroughNonTransient = 0b1000,
    /// Write-through non-transient, write-allocate
    WriteThroughNonTransientWriteAllocate = 0b1001,
    /// Write-through non-transient, read-allocate
    WriteThroughNonTransientReadAllocate = 0b1010,
    /// Write-through non-transient, read-write-allocate
    WriteThroughNonTransientReadWriteAllocate = 0b1011,
    /// Write-back non-transient, do not allocate
    WriteBackNonTransient = 0b1100,
    /// Write-back non-transient, write-allocate
    WriteBackNonTransientWriteAllocate = 0b1101,
    /// Write-back non-transient, read-allocate
    WriteBackNonTransientReadAllocate = 0b1110,
    /// Write-back non-transient, read-write-allocate
    WriteBackNonTransientReadWriteAllocate = 0b1111,
}

impl TryFrom<u8> for NormalMemory {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0b0001 => Ok(Self::WriteBackNonTransientWriteAllocate),
            0b0010 => Ok(Self::WriteThroughTransientReadAllocate),
            0b0011 => Ok(Self::WriteThroughTransientReadWriteAllocate),
            0b0100 => Ok(Self::NonCacheable),
            0b0101 => Ok(Self::WriteBackTransientWriteAllocate),
            0b0110 => Ok(Self::WriteBackTransientReadAllocate),
            0b0111 => Ok(Self::WriteBackTransientReadWriteAllocate),
            0b1000 => Ok(Self::WriteThroughNonTransient),
            0b1001 => Ok(Self::WriteThroughNonTransientWriteAllocate),
            0b1010 => Ok(Self::WriteThroughNonTransientReadAllocate),
            0b1011 => Ok(Self::WriteThroughNonTransientReadWriteAllocate),
            0b1100 => Ok(Self::WriteBackNonTransient),
            0b1101 => Ok(Self::WriteBackNonTransientWriteAllocate),
            0b1110 => Ok(Self::WriteBackNonTransientReadAllocate),
            0b1111 => Ok(Self::WriteBackNonTransientReadWriteAllocate),
            _ => Err(()),
        }
    }
}

impl NormalMemory {
    fn as_str(self) -> &'static str {
        match self {
            NormalMemory::WriteThroughTransientWriteAllocate => {
                "Write-through transient, write-allocate"
            }
            NormalMemory::WriteThroughTransientReadAllocate => {
                "Write-through transient, read-allocate"
            }
            NormalMemory::WriteThroughTransientReadWriteAllocate => {
                "Write-through transient, read-write-allocate"
            }
            NormalMemory::NonCacheable => "Non-Cacheable",
            NormalMemory::WriteBackTransientWriteAllocate => "Write-back transient, write-allocate",
            NormalMemory::WriteBackTransientReadAllocate => "Write-back transient, read-allocate",
            NormalMemory::WriteBackTransientReadWriteAllocate => {
                "Write-back transient, read-write-allocate"
            }
            NormalMemory::WriteThroughNonTransient => {
                "Write-through non-transient, do not allocate"
            }
            NormalMemory::WriteThroughNonTransientWriteAllocate => {
                "Write-through non-transient, write-allocate"
            }
            NormalMemory::WriteThroughNonTransientReadAllocate => {
                "Write-through non-transient, read-allocate"
            }
            NormalMemory::WriteThroughNonTransientReadWriteAllocate => {
                "Write-through non-transient, read-write-allocate"
            }
            NormalMemory::WriteBackNonTransient => "Write-back non-transient, do not allocate",
            NormalMemory::WriteBackNonTransientWriteAllocate => {
                "Write-back non-transient, write-allocate"
            }
            NormalMemory::WriteBackNonTransientReadAllocate => {
                "Write-back non-transient, read-allocate"
            }
            NormalMemory::WriteBackNonTransientReadWriteAllocate => {
                "Write-back non-transient, read-write-allocate"
            }
        }
    }
}

impl Display for NormalMemory {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn format_device() {
        assert_eq!(MairAttribute::DEVICE_NGNRNE.to_string(), "Device-nGnRnE");
        assert_eq!(MairAttribute::DEVICE_NGNRE.to_string(), "Device-nGnRE");
        assert_eq!(MairAttribute::DEVICE_NGRE.to_string(), "Device-nGRE");
        assert_eq!(MairAttribute::DEVICE_GRE.to_string(), "Device-GRE");
    }

    #[test]
    fn format_normal() {
        assert_eq!(
            MairAttribute::normal(
                NormalMemory::NonCacheable,
                NormalMemory::WriteBackNonTransient
            )
            .to_string(),
            "Normal, Inner Non-Cacheable, Outer Write-back non-transient, do not allocate"
        );
        assert_eq!(
            MairAttribute::normal(
                NormalMemory::WriteThroughTransientReadAllocate,
                NormalMemory::WriteBackNonTransientWriteAllocate
            )
            .to_string(),
            "Normal, Inner Write-through transient, read-allocate, Outer Write-back non-transient, write-allocate"
        );
        assert_eq!(
            MairAttribute::normal(
                NormalMemory::WriteThroughTransientReadWriteAllocate,
                NormalMemory::WriteThroughNonTransient
            )
            .to_string(),
            "Normal, Inner Write-through transient, read-write-allocate, Outer Write-through non-transient, do not allocate"
        );
    }

    #[test]
    fn format_unpredictable() {
        assert_eq!(
            MairAttribute(0b0000_0001).to_string(),
            "Unpredictable (0x01)"
        );
        assert_eq!(
            MairAttribute(0b0000_1111).to_string(),
            "Unpredictable (0x0f)"
        );
        assert_eq!(
            MairAttribute(0b0100_0000).to_string(),
            "Unpredictable (0x40)"
        );
        assert_eq!(
            MairAttribute(0b1111_0000).to_string(),
            "Unpredictable (0xf0)"
        );
    }

    #[test]
    fn format_mair() {
        assert_eq!(
            Mair(0x44ff04).to_string(),
            "{0: Device-nGnRE; \
            1: Normal, Inner Write-back non-transient, read-write-allocate, Outer Write-back non-transient, read-write-allocate; \
            2: Normal, Inner Non-Cacheable, Outer Non-Cacheable; \
            3: Device-nGnRnE; \
            4: Device-nGnRnE; \
            5: Device-nGnRnE; \
            6: Device-nGnRnE; \
            7: Device-nGnRnE}"
        );
    }

    #[test]
    fn build_const() {
        const MAIR: Mair = Mair::new([
            MairAttribute::DEVICE_NGNRE,
            MairAttribute::normal(NormalMemory::NonCacheable, NormalMemory::NonCacheable),
            MairAttribute::normal(
                NormalMemory::WriteBackNonTransientReadWriteAllocate,
                NormalMemory::WriteBackTransientReadWriteAllocate,
            ),
            MairAttribute::DEVICE_NGNRNE,
            MairAttribute::DEVICE_NGNRNE,
            MairAttribute::DEVICE_NGNRNE,
            MairAttribute::DEVICE_NGNRNE,
            MairAttribute::DEVICE_NGNRNE,
        ]);
        assert_eq!(MAIR.0, 0x0000_0000_007f_4404);

        const MAIR2: Mair = Mair::EMPTY
            .with_attribute(0, MairAttribute::DEVICE_NGNRE)
            .with_attribute(
                1,
                MairAttribute::normal(NormalMemory::NonCacheable, NormalMemory::NonCacheable),
            )
            .with_attribute(
                2,
                MairAttribute::normal(
                    NormalMemory::WriteBackNonTransientReadWriteAllocate,
                    NormalMemory::WriteBackTransientReadWriteAllocate,
                ),
            );
        assert_eq!(MAIR2, MAIR);
    }

    #[test]
    #[should_panic]
    fn invalid_index() {
        Mair::EMPTY.with_attribute(8, MairAttribute::DEVICE_GRE);
    }
}
