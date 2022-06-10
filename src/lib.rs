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
//! Full support is only provided for identity mapping; for other mapping schemes the user of the
//! library must implement some functionality themself including an implementation of the
//! [`Translation`](paging::Translation) trait.
//!
//! # Example
//!
//! ```
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
//!     Attributes::NORMAL | Attributes::NON_GLOBAL | Attributes::READ_ONLY,
//! );
//! // Set `TTBR0_EL1` to activate the page table.
//! # #[cfg(target_arch = "aarch64")]
//! idmap.activate();
//! ```

#![no_std]

pub mod idmap;
pub mod paging;

extern crate alloc;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AddressRangeError;
