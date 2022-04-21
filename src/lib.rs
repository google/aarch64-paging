// Copyright 2022 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

#![no_std]

pub mod idmap;
pub mod paging;

extern crate alloc;

#[macro_use]
extern crate bitflags;
