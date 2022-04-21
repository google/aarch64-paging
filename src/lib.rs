// SPDX-License-Identifier: GPL-2.0
// Copyright 2022 Google LLC
// Author: Ard Biesheuvel <ardb@google.com>

#![no_std]

pub mod idmap;
pub mod paging;

extern crate alloc;

#[macro_use]
extern crate bitflags;
