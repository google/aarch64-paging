// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! The randomized tests of `src/fuzz.rs`, but with page tables that are actually live, and with
//! checks of the effect of the updates on memory accesses.
//!
//! The tests apply random updates to page tables for the lower and the upper VA range, and check
//! the page tables against a model of the expected mappings after each update, like the tests on
//! the host. In addition, while the page tables are live, they access pages in the window of
//! virtual addresses that the updates apply to, before and after each update, and check that:
//!
//! - accesses to unmapped pages cause translation faults;
//! - accesses to pages whose access flag is clear cause access flag faults;
//! - loads from other pages return the tag of the physical page they should map, which is written
//!   to each physical page before it is first accessed via the page tables under test;
//! - stores to read-only pages cause permission faults, and stores to writable pages succeed.
//!
//! The accesses before each update make it likely that the TLB holds entries for the pages that the
//! update changes, so that if the update lacks TLB maintenance, the accesses after the update find
//! stale entries. Whether they do depends on the TLB, though, so this is not guaranteed, and
//! break-before-make violations or walk cache entries for freed tables are unlikely to be detected
//! at all. In particular, QEMU's TLB with TCG doesn't cache table descriptors, and it flushes all
//! entries for TLBI ASIDE1 and on ASID changes, and TLB maintenance takes effect without a DSB.

#[path = "../../src/fuzz/engine.rs"]
#[allow(
    dead_code,
    reason = "only the tests on the host use inputs provided by bolero"
)]
mod engine;

use crate::exceptions::{self, Fault};
use crate::{Config, DEVICE_ATTRIBUTES, DEVICES, RAM, RAM_ATTRIBUTES, console};
use aarch64_paging::Mapping;
use aarch64_paging::descriptor::{El1Attributes, PhysicalAddress};
use aarch64_paging::idmap::IdTranslation;
use aarch64_paging::paging::{
    Constraints, El1And0, El2And0, MemoryRegion, PAGE_SIZE, TranslationRegime, VaRange,
};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::arch::asm;
use core::fmt;
use engine::{
    BLOCK_PAGES, EL1_MEMORY_TYPES, EL1_OPTIONAL, HUGE_PAGES, Harness, Model, Op, PaSpace, Page,
    Rng, TestMapping, Variant, WINDOW_SIZE, Window,
};

/// The translation of the page tables that the tests operate on.
type TestTranslation<A> = IdTranslation<A>;

/// The physical memory that mappings are created in: the upper 3 GiB of RAM, as the program and
/// its heap are in the lowest GiB. This is just enough for the largest updates. The page tables
/// under test never map memory that doesn't exist, as speculative accesses to it might cause
/// external aborts on real hardware.
const PA_SPACE: PaSpace = PaSpace {
    base: 2 << 30,
    size: 3 << 30,
    wrap: true,
};

/// The window of the page tables for the lower VA range, which is above the identity map of the
/// devices and RAM.
const LOWER_WINDOW: Window = Window {
    base: 8 << 30,
    pages: WINDOW_SIZE / PAGE_SIZE,
};

/// A window at the top of the upper VA range, except for the last page, as the end of a
/// `MemoryRegion` that covers it would overflow.
const UPPER_WINDOW: Window = Window {
    base: WINDOW_SIZE.wrapping_neg(),
    pages: WINDOW_SIZE / PAGE_SIZE - 1,
};

/// The root level for 39-bit VA ranges, which is what TCR configures.
const ROOT_LEVEL: usize = 1;
const ASID: usize = 1;
/// The maximum number of valid descriptors that an update of live page tables may cover.
const LIVE_LIMIT: usize = 4096;

/// The top 16 bits of the tag of each physical page, the rest of which is its address.
const TAG: u64 = 0x7a67 << 48;

/// A translation regime that the tests use, depending on the exception level.
pub trait Regime:
    TranslationRegime<Attributes = El1Attributes, Asid = usize, VaRange = VaRange>
{
    const REGIME: Self;
    const NAME: &str;
}

impl Regime for El1And0 {
    const REGIME: Self = El1And0;
    const NAME: &str = "el1and0";
}

impl Regime for El2And0 {
    const REGIME: Self = El2And0;
    const NAME: &str = "el2and0";
}

fn new_lower<R: Regime>() -> TestMapping<R> {
    let mut mapping = Mapping::with_asid_and_va_range(
        IdTranslation::new(),
        ASID,
        ROOT_LEVEL,
        R::REGIME,
        VaRange::Lower,
    );
    // Map the devices and RAM exactly like the initial identity map, which these page tables
    // replace when they're activated.
    for ((start, end), attributes) in [(DEVICES, DEVICE_ATTRIBUTES), (RAM, RAM_ATTRIBUTES)] {
        mapping
            .map_range(
                &MemoryRegion::new(start, end),
                PhysicalAddress(start),
                attributes,
                Constraints::empty(),
            )
            .unwrap();
    }
    mapping
}

fn new_upper<R: Regime>() -> TestMapping<R> {
    Mapping::with_asid_and_va_range(
        IdTranslation::new(),
        ASID,
        ROOT_LEVEL,
        R::REGIME,
        VaRange::Upper,
    )
}

fn variant<R: Regime>(new_mapping: fn() -> TestMapping<R>, window: Window) -> Variant<R> {
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

/// Returns what `page` is expected to map.
fn expected(model: &Model<El1Attributes>, page: usize) -> Option<Page<El1Attributes>> {
    let run = &model.runs[model.runs.partition_point(|run| run.end <= page)];
    run.page.map(|p| Page {
        pa: p.pa + (page - run.start) * PAGE_SIZE,
        ..p
    })
}

/// Describes the outcome of a load.
fn describe(result: Result<u64, Fault>) -> String {
    match result {
        Ok(value) if value & !((1 << 48) - 1) == TAG => {
            format!("returned the tag of PA {:#x}", value & ((1 << 48) - 1))
        }
        Ok(value) => format!("returned {value:#x}, which is not a tag"),
        Err(fault) => format!("caused a {fault}"),
    }
}

#[derive(Default)]
struct Stats {
    loads: usize,
    stores: usize,
    faults: usize,
}

/// Accesses pages via the page tables under test, and checks the outcome.
struct Probes {
    rng: Rng<'static>,
    /// The pages of `PA_SPACE` that have been tagged, one bit per page.
    tagged: Vec<u64>,
    /// The pages to access before and after the current update.
    pages: Vec<usize>,
    /// The number of the current update in the sequence.
    op: usize,
    stats: Stats,
}

impl Probes {
    /// Writes the tag of the physical page at `pa` to it, unless that was done before, and
    /// returns the tag.
    fn tag(&mut self, pa: usize) -> u64 {
        let tag = TAG | pa as u64;
        let index = (pa - PA_SPACE.base) / PAGE_SIZE;
        let (word, bit) = (index / 64, 1 << (index % 64));
        if self.tagged[word] & bit == 0 {
            // SAFETY: `pa` is identity mapped, and the program doesn't use this memory otherwise.
            unsafe {
                (pa as *mut u64).write_volatile(tag);
                // Make the tag visible to non-cacheable accesses as well.
                asm!("dc civac, {}", "dsb sy", in(reg) pa, options(nostack));
            }
            self.tagged[word] |= bit;
        }
        tag
    }

    /// Returns the pages to access for `op`: those in the range that it updates, or a sample of
    /// them if there are many, and those right outside of it.
    fn op_pages(&mut self, window: &Window, op: &Op<El1Attributes>) -> Vec<usize> {
        let (start, end) = match *op {
            Op::Map { start, end, .. } | Op::Modify { start, end, .. } => (start, end),
            _ => return Vec::new(),
        };
        let mut pages = Vec::new();
        if end - start <= 32 {
            pages.extend(start..end);
        } else {
            pages.extend(start..start + 4);
            pages.extend(end - 4..end);
            // The pages on either side of block boundaries.
            for granule in [HUGE_PAGES, BLOCK_PAGES] {
                let mut boundary = (start + 1).next_multiple_of(granule);
                while boundary < end && pages.len() < 40 {
                    pages.extend([boundary - 1, boundary]);
                    boundary += granule;
                }
            }
            for _ in 0..16 {
                pages.push(start + self.rng.below(end - start));
            }
        }
        if start > 0 {
            pages.push(start - 1);
        }
        if end < window.pages {
            pages.push(end);
        }
        pages
    }

    /// Returns random pages, which are likely to be mapped.
    fn random_pages(&mut self, window: &Window, model: &Model<El1Attributes>) -> Vec<usize> {
        let mut pages = Vec::new();
        // Most updates are near the start of the window.
        for _ in 0..4 {
            pages.push(self.rng.below(4 * BLOCK_PAGES));
            pages.push(self.rng.below(window.pages));
        }
        let mapped = model
            .runs
            .iter()
            .filter(|run| run.page.is_some())
            .collect::<Vec<_>>();
        if !mapped.is_empty() {
            for _ in 0..8 {
                let run = mapped[self.rng.below(mapped.len())];
                pages.push(run.start + self.rng.below(run.end - run.start));
            }
        }
        pages
    }

    /// Accesses `page`, and checks that the outcome matches what `model` says it maps.
    fn probe<R: Regime>(
        &mut self,
        variant: &Variant<R>,
        model: &Model<El1Attributes>,
        page: usize,
        when: &str,
    ) {
        let va = variant.window.va(page);
        let expected = expected(model, page);
        let pa = expected.map(|p| variant.pa.wrap(p.pa));
        let tag = pa.map(|pa| self.tag(pa));
        let fail = |access: &str, outcome: String| -> ! {
            let what = match (expected, pa) {
                (Some(p), Some(pa)) => format!("map PA {pa:#x} with {:?}", p.flags),
                _ => String::from("be unmapped"),
            };
            panic!(
                "{when} update {}: {access} of page {page:#x} at {va:#x} {outcome}, but the page \
                 should {what}",
                self.op
            )
        };

        let expected_load = match expected {
            None => Err(Fault::Translation),
            Some(p) if !p.flags.contains(El1Attributes::ACCESSED) => Err(Fault::AccessFlag),
            Some(_) => Ok(tag.unwrap()),
        };
        // SAFETY: The window is not used by the program otherwise, and the page tables under test
        // only map memory in `PA_SPACE` there, which the program only accesses to write the tags.
        let load = unsafe { exceptions::read(va) };
        if load != expected_load {
            fail("load", describe(load));
        }
        self.stats.loads += 1;
        self.stats.faults += usize::from(load.is_err());

        if let (Some(p), Some(tag)) = (expected, tag)
            && p.flags.contains(El1Attributes::ACCESSED)
        {
            let expected_store = if p.flags.contains(El1Attributes::READ_ONLY) {
                Err(Fault::Permission)
            } else {
                Ok(())
            };
            // SAFETY: As above, and the store doesn't change the tag.
            let store = unsafe { exceptions::write(va, tag) };
            if store != expected_store {
                let outcome = match store {
                    Ok(()) => String::from("succeeded"),
                    Err(fault) => format!("caused a {fault}"),
                };
                fail("store", outcome);
            }
            self.stats.stores += 1;
            self.stats.faults += usize::from(store.is_err());
        }
    }
}

impl<R: Regime> Harness<R> for Probes {
    fn log(&mut self, line: fmt::Arguments) {
        console::log(line);
    }

    fn before(
        &mut self,
        variant: &Variant<R>,
        mapping: &TestMapping<R>,
        model: &Model<El1Attributes>,
        op: &Op<El1Attributes>,
    ) {
        self.pages = self.op_pages(&variant.window, op);
        if mapping.active() {
            // Make it likely that the TLB holds entries for the pages that the update may change.
            for page in core::mem::take(&mut self.pages) {
                self.probe(variant, model, page, "Before");
                self.pages.push(page);
            }
        }
    }

    fn after(
        &mut self,
        variant: &Variant<R>,
        mapping: &TestMapping<R>,
        model: &Model<El1Attributes>,
        _op: &Op<El1Attributes>,
    ) {
        if mapping.active() {
            let pages = core::mem::take(&mut self.pages);
            let random = self.random_pages(&variant.window, model);
            for page in pages.into_iter().chain(random) {
                self.probe(variant, model, page, "After");
            }
        }
        self.op += 1;
    }
}

/// Runs the tests with translation regime `R`, and returns the number of sequences that passed.
pub fn run<R: Regime>(config: &Config) -> usize {
    console::set_verbose(config.verbose);
    let mut probes = Probes {
        rng: Rng::SplitMix(0),
        tagged: vec![0; PA_SPACE.size / PAGE_SIZE / 64],
        pages: Vec::new(),
        op: 0,
        stats: Stats::default(),
    };
    let mut sequences = 0;
    for (range, enabled, upper) in [
        ("lower", config.lower, false),
        ("upper", config.upper, true),
    ] {
        if !enabled {
            continue;
        }
        let variant = if upper {
            variant(new_upper::<R>, UPPER_WINDOW)
        } else {
            variant(new_lower::<R>, LOWER_WINDOW)
        };
        // The page tables for the upper VA range use the ASID in TTBR1.
        crate::set_tcr_a1(upper);
        let name = format!("vm_{}_{range}", R::NAME);
        let seeds = match config.seed {
            Some(seed) => vec![seed],
            None => {
                let first = engine::first_seed(&name);
                (0..config.runs as u64)
                    .map(|i| first.wrapping_add(i))
                    .collect()
            }
        };
        for seed in seeds {
            println!("{name}: seed {seed:#x}, {} updates", config.ops);
            probes.rng = Rng::SplitMix(seed ^ 0x7072_6f62_6573);
            probes.op = 0;
            engine::run(&variant, &mut Rng::SplitMix(seed), config.ops, &mut probes);
            sequences += 1;
        }
        crate::set_tcr_a1(false);
    }
    let stats = &probes.stats;
    println!(
        "{} loads and {} stores via the page tables under test, {} of which faulted as expected",
        stats.loads, stats.stores, stats.faults
    );
    sequences
}
