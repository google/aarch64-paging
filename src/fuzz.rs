// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Randomized tests that apply long sequences of random updates to page tables via [`Mapping`],
//! both while they are live and while they are not, and check after each update that:
//!
//! - the page tables map exactly what a simple model of the expected mappings says they should,
//!   which in particular means that failed updates leave the page tables unmodified, unless they
//!   are documented to fail halfway;
//! - updates of memory that is not mapped are never refused, even while the page tables are live;
//! - the TLB emulation detects no TLB conflicts, stale TLB entries or prematurely freed tables.
//!
//! Each test runs a few sequences with fixed seeds by default. Set `AARCH64_PAGING_FUZZ_RUNS` to
//! run more sequences, `AARCH64_PAGING_FUZZ_OPS` to change the number of updates per sequence, and
//! `AARCH64_PAGING_FUZZ_SEED` to run only the sequence with the given seed, e.g., to reproduce a
//! failure, as the seed of each sequence is printed before it starts.
//!
//! In addition, `fuzz_inputs` applies sequences of updates that are described by inputs provided by
//! [bolero](https://github.com/camshaft/bolero), which determine the choices that are otherwise
//! made by the pseudo-random number generator. Under `cargo test`, these are the inputs that earlier
//! fuzzing sessions saved in `src/__fuzz__/fuzz__inputs`, if any, starting with those that failed,
//! followed by random inputs, for up to a second in total unless `BOLERO_RANDOM_TEST_TIME_MS` is
//! set. If a random input fails, the `BOLERO_RANDOM_SEED` to reproduce it is printed. For
//! coverage-guided fuzzing with libFuzzer, install `cargo-bolero` and run
//!
//! ```text
//! cargo bolero test fuzz::fuzz_inputs --sanitizer NONE
//! ```
//!
//! or omit `--sanitizer NONE` on a nightly toolchain to use AddressSanitizer as well. Inputs that
//! fail are saved in `src/__fuzz__/fuzz__inputs/crashes`, from where `cargo test` replays them. Set
//! `AARCH64_PAGING_FUZZ_VERBOSE` to print the updates that an input describes.
//!
//! The generation of the updates and the checks against the model are in [`engine`], which is
//! shared with the tests in `vmtests/` that apply the updates to page tables that are actually
//! live in a virtual machine, and check the effect of each update on memory accesses.

mod engine;

use crate::descriptor::{El1Attributes, Stage2Attributes};
use crate::idmap::IdTranslation;
use crate::paging::{El1And0, El2And0, PAGE_SIZE, Stage2, TranslationRegime, VaRange};
use crate::tlb::{self, Emulated, Tracked};
use core::fmt;
use engine::{
    EL1_MEMORY_TYPES, EL1_OPTIONAL, Harness, Model, Op, PaSpace, Rng, TestMapping, Variant,
    WINDOW_SIZE, Window,
};
use std::{env, println, vec};

/// The translation of the page tables that the tests operate on, which reports their translation
/// tables to the TLB emulation.
type TestTranslation<A> = Tracked<IdTranslation<A>>;

const DEFAULT_RUNS: usize = 2;
const DEFAULT_OPS: usize = 300;
/// Physical addresses are chosen below this limit, and stay well within 48 bits when moved.
const PA_SPACE: PaSpace = PaSpace {
    base: 0,
    size: 1 << 40,
    wrap: false,
};
/// The maximum number of valid descriptors that an update of a live page table may cover, as the
/// cost of the TLB emulation grows quadratically with the number of TLB entries.
const LIVE_LIMIT: usize = 512;

/// Checks the TLB emulation after each update, and prints the updates if `verbose` is set.
struct Emulation {
    verbose: bool,
}

impl<R: TranslationRegime> Harness<R> for Emulation {
    fn log(&mut self, line: fmt::Arguments) {
        if self.verbose {
            println!("{line}");
        }
    }

    fn after(
        &mut self,
        variant: &Variant<R>,
        mapping: &TestMapping<R>,
        _model: &Model<R::Attributes>,
        _op: &Op<R::Attributes>,
    ) {
        // Check the TLB emulation for stale entries anywhere in the window, also after updates
        // that failed, rather than just in the updated range after successful updates.
        tlb::check_coherency(
            mapping.root_address(),
            &variant.window.region(0, variant.window.pages),
        );
    }
}

fn env_var(name: &str) -> Option<u64> {
    let value = env::var(name).ok()?;
    let parsed = match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    };
    Some(parsed.unwrap_or_else(|_| panic!("Invalid value for {name}: {value}")))
}

/// Runs the randomized test for `variant`, with seeds that are derived from `name` unless a seed
/// is given in the environment.
fn fuzz<R: TranslationRegime>(name: &str, variant: Variant<R>) {
    let ops = env_var("AARCH64_PAGING_FUZZ_OPS").map_or(DEFAULT_OPS, |n| n as usize);
    let seeds = match env_var("AARCH64_PAGING_FUZZ_SEED") {
        Some(seed) => vec![seed],
        None => {
            let first = engine::first_seed(name);
            let runs = env_var("AARCH64_PAGING_FUZZ_RUNS").map_or(DEFAULT_RUNS, |n| n as usize);
            (0..runs as u64).map(|i| first.wrapping_add(i)).collect()
        }
    };
    for seed in seeds {
        println!("Seed {seed:#x}, {ops} operations");
        engine::run(
            &variant,
            &mut Rng::SplitMix(seed),
            ops,
            &mut Emulation { verbose: true },
        );
    }
}

const LOWER_WINDOW: Window = Window {
    base: 0,
    pages: WINDOW_SIZE / PAGE_SIZE,
};

/// A window at the top of the upper VA range, except for the last page, as the end of a
/// `MemoryRegion` that covers it would overflow.
const UPPER_WINDOW: Window = Window {
    base: WINDOW_SIZE.wrapping_neg(),
    pages: WINDOW_SIZE / PAGE_SIZE - 1,
};

fn el1_variant(
    new_mapping: fn() -> TestMapping<Emulated<El1And0>>,
    window: Window,
) -> Variant<Emulated<El1And0>> {
    Variant {
        new_mapping,
        window,
        base: El1Attributes::VALID.union(El1Attributes::ACCESSED),
        memory_types: EL1_MEMORY_TYPES,
        optional: EL1_OPTIONAL,
        software: El1Attributes::SWFLAG_1,
        pa: PA_SPACE,
        live_limit: LIVE_LIMIT,
    }
}

fn el1and0() -> Variant<Emulated<El1And0>> {
    el1_variant(
        || tlb::mapping_with_asid(IdTranslation::new(), 1, 1, El1And0, VaRange::Lower).into_inner(),
        LOWER_WINDOW,
    )
}

fn el1and0_upper() -> Variant<Emulated<El1And0>> {
    el1_variant(
        || tlb::mapping_with_asid(IdTranslation::new(), 1, 1, El1And0, VaRange::Upper).into_inner(),
        UPPER_WINDOW,
    )
}

/// Attributes that may be changed on live EL2&0 mappings, which are all non-global.
const EL2AND0_OPTIONAL: &[El1Attributes] = &[
    El1Attributes::READ_ONLY,
    El1Attributes::ACCESSED,
    El1Attributes::PXN,
    El1Attributes::UXN,
    El1Attributes::SWFLAG_0,
    El1Attributes::USER,
];

fn el2and0_non_global() -> Variant<Emulated<El2And0>> {
    Variant {
        new_mapping: || {
            tlb::mapping_with_asid(IdTranslation::new(), 2, 1, El2And0, VaRange::Lower).into_inner()
        },
        window: LOWER_WINDOW,
        base: El1Attributes::VALID
            .union(El1Attributes::ACCESSED)
            .union(El1Attributes::NON_GLOBAL),
        memory_types: EL1_MEMORY_TYPES,
        optional: EL2AND0_OPTIONAL,
        software: El1Attributes::SWFLAG_1,
        pa: PA_SPACE,
        live_limit: LIVE_LIMIT,
    }
}

const STAGE2_MEMORY_TYPES: &[Stage2Attributes] = &[
    Stage2Attributes::MEMATTR_NORMAL_INNER_WB
        .union(Stage2Attributes::MEMATTR_NORMAL_OUTER_WB)
        .union(Stage2Attributes::SH_INNER),
    Stage2Attributes::MEMATTR_NORMAL_INNER_NC.union(Stage2Attributes::MEMATTR_NORMAL_OUTER_NC),
    Stage2Attributes::MEMATTR_DEVICE_nGnRE,
];

/// Attributes that may all be changed on live stage 2 mappings.
const STAGE2_OPTIONAL: &[Stage2Attributes] = &[
    Stage2Attributes::S2AP_ACCESS_RO,
    Stage2Attributes::S2AP_ACCESS_WO,
    Stage2Attributes::ACCESS_FLAG,
    Stage2Attributes::XN,
    Stage2Attributes::SWFLAG_0,
];

fn stage2() -> Variant<Emulated<Stage2>> {
    Variant {
        new_mapping: || tlb::mapping(IdTranslation::new(), 1, Stage2).into_inner(),
        window: LOWER_WINDOW,
        base: Stage2Attributes::VALID.union(Stage2Attributes::ACCESS_FLAG),
        memory_types: STAGE2_MEMORY_TYPES,
        optional: STAGE2_OPTIONAL,
        software: Stage2Attributes::SWFLAG_1,
        pa: PA_SPACE,
        live_limit: LIVE_LIMIT,
    }
}

#[test]
fn fuzz_el1and0() {
    fuzz("fuzz_el1and0", el1and0());
}

#[test]
fn fuzz_el1and0_upper() {
    fuzz("fuzz_el1and0_upper", el1and0_upper());
}

#[test]
fn fuzz_el2and0_non_global() {
    fuzz("fuzz_el2and0_non_global", el2and0_non_global());
}

#[test]
fn fuzz_stage2() {
    fuzz("fuzz_stage2", stage2());
}

/// The maximum number of updates that an input from bolero may describe, which keeps the time spent
/// on each input short.
const MAX_FUZZ_OPS: usize = 64;

/// Applies the sequence of updates described by `data`, an input provided by bolero, to a page
/// table, using the translation regime and VA range selected by its first byte, and checks the page
/// table after each update. The updates are printed if `AARCH64_PAGING_FUZZ_VERBOSE` is set.
fn fuzz_one(data: &[u8]) {
    let Some((&variant, data)) = data.split_first() else {
        return;
    };
    let rng = &mut Rng::Input(data);
    let harness = &mut Emulation {
        verbose: env::var_os("AARCH64_PAGING_FUZZ_VERBOSE").is_some(),
    };
    match variant % 4 {
        0 => engine::run(&el1and0(), rng, MAX_FUZZ_OPS, harness),
        1 => engine::run(&el1and0_upper(), rng, MAX_FUZZ_OPS, harness),
        2 => engine::run(&el2and0_non_global(), rng, MAX_FUZZ_OPS, harness),
        _ => engine::run(&stage2(), rng, MAX_FUZZ_OPS, harness),
    }
}

/// Applies the sequences of updates described by inputs provided by bolero, see the module
/// documentation.
#[test]
fn fuzz_inputs() {
    bolero::check!().for_each(fuzz_one);
}
