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

use crate::descriptor::{
    El1Attributes, PagingAttributes, PhysicalAddress, Stage2Attributes, UpdatableDescriptor,
    VirtualAddress,
};
use crate::idmap::IdTranslation;
use crate::paging::{
    Constraints, El1And0, El2And0, MemoryRegion, PAGE_SIZE, Stage2, TranslationRegime, VaRange,
    granularity_at_level,
};
use crate::tlb::{self, Emulated};
use crate::{MapError, Mapping};
use bitflags::Flags;
use core::cell::Cell;
use std::vec::Vec;
use std::{env, println, vec};

const DEFAULT_RUNS: usize = 2;
const DEFAULT_OPS: usize = 300;

/// The number of pages mapped by a level 2 block descriptor.
const BLOCK_PAGES: usize = 1 << 9;
/// The number of pages mapped by a level 1 block descriptor.
const HUGE_PAGES: usize = 1 << 18;
/// The size of the virtual address range that the tests operate on.
const WINDOW_SIZE: usize = 4 << 30;
/// Physical addresses are chosen below this limit, and stay well within 48 bits when moved.
const PA_LIMIT: usize = 1 << 40;
/// The maximum number of valid descriptors that an update of a live page table may cover, as the
/// cost of the TLB emulation grows quadratically with the number of TLB entries.
const LIVE_LIMIT: usize = 512;

/// The source of the choices that determine a sequence of updates: either a small, deterministic
/// pseudo-random number generator (SplitMix64), or an input provided by bolero.
enum Rng<'a> {
    SplitMix(u64),
    Input(&'a [u8]),
}

impl Rng<'_> {
    /// Returns a number in `0..n`.
    ///
    /// For an input, this consumes as few of its bytes as possible, so that a fuzzer that mutates a
    /// byte mostly affects a single choice, and it returns 0 once the input is used up.
    fn below(&mut self, n: usize) -> usize {
        match self {
            Self::SplitMix(state) => {
                *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = *state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z ^ (z >> 31)) % n as u64) as usize
            }
            Self::Input(data) => {
                let mut value = 0;
                let mut range = 1;
                while range < n {
                    let (&byte, rest) = data.split_first().unwrap_or((&0, &[]));
                    *data = rest;
                    value = (value << 8) | usize::from(byte);
                    range = range.saturating_mul(256);
                }
                value % n.max(1)
            }
        }
    }

    /// Returns whether an input has been used up.
    fn exhausted(&self) -> bool {
        matches!(self, Self::Input(data) if data.is_empty())
    }

    /// Returns true with the given probability, in percent.
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn choose<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }

    /// Returns the union of a random subset of `items`.
    fn subset<A: PagingAttributes>(&mut self, items: &[A]) -> A {
        items
            .iter()
            .filter(|_| self.chance(30))
            .fold(A::empty(), |acc, &item| acc | item)
    }
}

/// The range of virtual addresses that a test operates on, which is addressed in pages.
#[derive(Clone, Copy)]
struct Window {
    base: usize,
    pages: usize,
}

impl Window {
    fn va(&self, page: usize) -> usize {
        self.base.wrapping_add(page * PAGE_SIZE)
    }

    fn page(&self, va: VirtualAddress) -> usize {
        va.0.wrapping_sub(self.base) / PAGE_SIZE
    }

    fn region(&self, start: usize, end: usize) -> MemoryRegion {
        MemoryRegion::new(self.va(start), self.va(end))
    }
}

/// What a page is expected to map.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Page<A> {
    pa: usize,
    flags: A,
}

/// A run of consecutive pages that are either all unmapped, or all mapped with the same attributes
/// to consecutive physical pages. `page` describes the first page of the run.
#[derive(Clone, Copy, Debug)]
struct Run<A> {
    start: usize,
    end: usize,
    page: Option<Page<A>>,
}

/// A model of what each page of the window is expected to map.
struct Model<A> {
    runs: Vec<Run<A>>,
}

impl<A: PagingAttributes> Model<A> {
    fn new(pages: usize) -> Self {
        Self {
            runs: vec![Run {
                start: 0,
                end: pages,
                page: None,
            }],
        }
    }

    /// Makes sure that a run starts at `page`.
    fn split(&mut self, page: usize) {
        if let Some(i) = self
            .runs
            .iter()
            .position(|run| run.start < page && page < run.end)
        {
            let run = self.runs[i];
            self.runs[i].end = page;
            self.runs.insert(
                i + 1,
                Run {
                    start: page,
                    end: run.end,
                    page: run.page.map(|p| Page {
                        pa: p.pa + (page - run.start) * PAGE_SIZE,
                        ..p
                    }),
                },
            );
        }
    }

    /// Replaces what the pages in `start..end` are expected to map by the result of `f`, which is
    /// given the first page of each run in that range and what it maps.
    fn update(
        &mut self,
        start: usize,
        end: usize,
        f: impl Fn(usize, Option<Page<A>>) -> Option<Page<A>>,
    ) {
        if start >= end {
            return;
        }
        self.split(start);
        self.split(end);
        for run in &mut self.runs {
            if start <= run.start && run.end <= end {
                run.page = f(run.start, run.page);
            }
        }
        // Merge adjacent runs that map consecutive pages in the same way.
        let mut merged: Vec<Run<A>> = Vec::with_capacity(self.runs.len());
        for run in self.runs.drain(..) {
            if let Some(last) = merged.last_mut() {
                let contiguous = match (last.page, run.page) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        a.flags == b.flags && a.pa + (last.end - last.start) * PAGE_SIZE == b.pa
                    }
                    _ => false,
                };
                if contiguous {
                    last.end = run.end;
                    continue;
                }
            }
            merged.push(run);
        }
        self.runs = merged;
    }

    fn is_unmapped(&self, start: usize, end: usize) -> bool {
        self.runs
            .iter()
            .filter(|run| run.start < end && start < run.end)
            .all(|run| run.page.is_none())
    }
}

/// A change made to each valid descriptor in a range by `modify_range`.
#[derive(Clone, Copy, Debug)]
enum Change<A> {
    /// Sets and clears the given attributes.
    Flags { set: A, clear: A },
    /// Clears the valid bit.
    Unmap,
    /// Moves the output address by the given offset, which is a multiple of the size of a level 1
    /// block, so that it is suitably aligned for descriptors at any level.
    Move { offset: usize },
    /// Like `Flags`, but fails at the first valid descriptor that starts at or after `page`.
    FailAt { page: usize, set: A, clear: A },
    /// Like `Flags`, but fails at the first valid descriptor after the updater has been called
    /// `calls` times. As `modify_range` calls the updater for each descriptor twice if the page
    /// tables are live, once to check for break-before-make violations and once to apply the
    /// change, this may fail halfway through applying the change.
    FailAfter { calls: usize, set: A, clear: A },
}

/// A way to call `map_range` or `modify_range` that must fail without making any changes.
#[derive(Clone, Copy, Debug)]
enum Invalid<A> {
    /// The range ends before it starts.
    Backwards { start: usize, end: usize },
    /// The range extends beyond the virtual address range of the page tables.
    OutOfRange,
    /// The attributes include `TABLE_OR_PAGE`, which `map_range` doesn't accept.
    TableOrPage { start: usize, end: usize, flags: A },
}

#[derive(Clone, Copy, Debug)]
enum Op<A> {
    Map {
        start: usize,
        end: usize,
        pa: usize,
        flags: A,
        constraints: Constraints,
    },
    Modify {
        start: usize,
        end: usize,
        change: Change<A>,
    },
    Invalid(Invalid<A>),
    Compact,
    Activate,
    Deactivate,
}

/// The configuration of the test for a translation regime.
struct Variant<R: TranslationRegime> {
    new_mapping: fn() -> Mapping<IdTranslation<R::Attributes>, R>,
    window: Window,
    /// The attributes that every valid mapping has.
    base: R::Attributes,
    /// The memory types that mappings may have, including the shareability, which can only be
    /// changed using break-before-make.
    memory_types: &'static [R::Attributes],
    /// Other attributes that are set or cleared at random, some of which can only be changed using
    /// break-before-make.
    optional: &'static [R::Attributes],
    /// A software attribute that is sometimes set in invalid descriptors.
    software: R::Attributes,
}

impl<R: TranslationRegime> Variant<R> {
    /// Returns a random range of pages, biased towards small ranges at the start of the window,
    /// and towards ranges that are aligned to block boundaries.
    fn random_range(&self, rng: &mut Rng<'_>) -> (usize, usize) {
        let pages = self.window.pages;
        let (granule, count) = match rng.below(10) {
            0..6 => (1, 1 + rng.below(32)),
            6..9 => (BLOCK_PAGES, 1 + rng.below(3)),
            _ => (HUGE_PAGES, 1 + rng.below(2)),
        };
        let limit = if rng.chance(70) {
            (4 * BLOCK_PAGES).max(granule)
        } else {
            pages
        };
        let mut start = rng.below(limit) / granule * granule;
        let mut end = start + granule * count;
        if granule > 1 && rng.chance(25) {
            // Make the range end or start halfway through a block.
            if rng.chance(50) {
                start += rng.below(BLOCK_PAGES);
            } else {
                end -= 1 + rng.below(BLOCK_PAGES - 1);
            }
        }
        (start.min(pages - 1), end.min(pages))
    }

    /// Returns random attributes for a valid mapping.
    fn random_flags(&self, rng: &mut Rng<'_>) -> R::Attributes {
        self.base | rng.choose(self.memory_types) | rng.subset(self.optional)
    }

    fn random_change(&self, rng: &mut Rng<'_>, start: usize, end: usize) -> Change<R::Attributes> {
        let set = rng.subset(self.optional);
        let clear = rng.subset(self.optional) - set;
        match rng.below(10) {
            0..4 => Change::Flags { set, clear },
            4 => {
                // Change the memory type.
                let memory_type = rng.choose(self.memory_types);
                let all = self
                    .memory_types
                    .iter()
                    .fold(R::Attributes::empty(), |acc, &t| acc | t);
                Change::Flags {
                    set: memory_type,
                    clear: all & !memory_type,
                }
            }
            5 => Change::Unmap,
            6 => Change::Move {
                offset: (1 + rng.below(8)) * HUGE_PAGES * PAGE_SIZE,
            },
            7 | 8 => Change::FailAt {
                page: start + rng.below(end - start),
                set,
                clear,
            },
            _ => Change::FailAfter {
                calls: rng.below(8),
                set,
                clear,
            },
        }
    }

    fn random_op(&self, rng: &mut Rng<'_>, live: bool) -> Op<R::Attributes> {
        let (start, end) = self.random_range(rng);
        match rng.below(100) {
            0..40 => {
                let (flags, constraints) = if rng.chance(80) {
                    let constraints = if end - start <= BLOCK_PAGES && rng.chance(20) {
                        Constraints::NO_BLOCK_MAPPINGS
                    } else {
                        Constraints::empty()
                    };
                    (self.random_flags(rng), constraints)
                } else {
                    // Unmap the range, leaving a software attribute set in some descriptors.
                    let software = if rng.chance(20) {
                        self.software
                    } else {
                        R::Attributes::empty()
                    };
                    (software, Constraints::empty())
                };
                // Use a physical address that permits block mappings in most cases, and always for
                // ranges larger than a block, which would need too many page mappings otherwise.
                let pa = if end - start > BLOCK_PAGES || rng.chance(80) {
                    let va = self.window.va(start);
                    rng.below(PA_LIMIT / (HUGE_PAGES * PAGE_SIZE)) * HUGE_PAGES * PAGE_SIZE
                        + va % (HUGE_PAGES * PAGE_SIZE)
                } else {
                    rng.below(PA_LIMIT / PAGE_SIZE) * PAGE_SIZE
                };
                Op::Map {
                    start,
                    end,
                    pa,
                    flags,
                    constraints,
                }
            }
            40..75 => Op::Modify {
                start,
                end,
                change: self.random_change(rng, start, end),
            },
            75..80 => Op::Compact,
            80..95 => {
                if live {
                    Op::Deactivate
                } else {
                    Op::Activate
                }
            }
            _ => Op::Invalid(match rng.below(3) {
                0 if start + 1 < end => Invalid::Backwards { start, end },
                1 => Invalid::OutOfRange,
                _ => Invalid::TableOrPage {
                    start,
                    end,
                    flags: self.random_flags(rng) | R::Attributes::TABLE_OR_PAGE,
                },
            }),
        }
    }
}

impl<A: Copy> Op<A> {
    /// Shrinks the range of a `Map` or `Modify` operation from the end until `fits` returns true for
    /// it, or until it covers a single page.
    fn shrink(self, fits: impl Fn(usize, usize) -> bool) -> Self {
        let shrink = |start: usize, mut end: usize| {
            while end - start > 1 && !fits(start, end) {
                end = start + (end - start) / 2;
            }
            end
        };
        match self {
            Op::Map {
                start,
                end,
                pa,
                flags,
                constraints,
            } => Op::Map {
                start,
                end: shrink(start, end),
                pa,
                flags,
                constraints,
            },
            Op::Modify { start, end, change } => {
                let end = shrink(start, end);
                let change = match change {
                    Change::FailAt { page, set, clear } => Change::FailAt {
                        page: page.min(end - 1),
                        set,
                        clear,
                    },
                    change => change,
                };
                Op::Modify { start, end, change }
            }
            op => op,
        }
    }
}

/// Returns the number of block, page and invalid descriptors that map the pages in `start..end`,
/// and how many of those are valid.
fn count_descriptors<R: TranslationRegime>(
    mapping: &Mapping<IdTranslation<R::Attributes>, R>,
    window: &Window,
    start: usize,
    end: usize,
) -> (usize, usize) {
    let (mut all, mut valid) = (0, 0);
    mapping
        .walk_range(&window.region(start, end), &mut |_, desc, _| {
            all += 1;
            valid += usize::from(desc.is_valid());
            Ok(())
        })
        .unwrap();
    (all, valid)
}

/// Checks that the page tables map exactly what the model says they should.
fn check<R: TranslationRegime>(
    mapping: &Mapping<IdTranslation<R::Attributes>, R>,
    model: &Model<R::Attributes>,
    window: &Window,
) {
    let mut runs = model.runs.iter().peekable();
    let mut next = 0;
    mapping
        .walk_range(
            &window.region(0, window.pages),
            &mut |region, desc, level| {
                let (first, last) = (window.page(region.start()), window.page(region.end()));
                assert_eq!(first, next, "walk_range skipped pages");
                next = last;
                let mut page = first;
                while page < last {
                    let run = runs.peek().expect("model doesn't cover the window");
                    if run.end <= page {
                        runs.next();
                        continue;
                    }
                    let expected = run.page.map(|p| Page {
                        pa: p.pa + (page - run.start) * PAGE_SIZE,
                        flags: p.flags - R::Attributes::TABLE_OR_PAGE,
                    });
                    let actual = desc.is_valid().then(|| {
                        let va = window.va(page);
                        Page {
                            pa: desc.output_address().0 + va % granularity_at_level(level),
                            flags: desc.flags() - R::Attributes::TABLE_OR_PAGE,
                        }
                    });
                    assert_eq!(
                        actual,
                        expected,
                        "page at {:#x} (level {level} descriptor {desc:?})",
                        window.va(page)
                    );
                    page = run.end.min(last);
                }
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(next, window.pages, "walk_range didn't cover the window");
}

fn apply<R: TranslationRegime>(
    mapping: &mut Mapping<IdTranslation<R::Attributes>, R>,
    model: &mut Model<R::Attributes>,
    window: &Window,
    ttbr: &mut Option<usize>,
    op: Op<R::Attributes>,
) {
    let live = ttbr.is_some();
    match op {
        Op::Map {
            start,
            end,
            pa,
            flags,
            constraints,
        } => {
            let unmapped = model.is_unmapped(start, end);
            match mapping.map_range(
                &window.region(start, end),
                PhysicalAddress(pa),
                flags,
                constraints,
            ) {
                Ok(()) => model.update(start, end, |run, _| {
                    flags.contains(R::Attributes::VALID).then_some(Page {
                        pa: pa + (run - start) * PAGE_SIZE,
                        flags,
                    })
                }),
                // Updates of live mappings may be refused, but not if nothing is mapped yet.
                Err(MapError::BreakBeforeMakeViolation(_)) if live && !unmapped => {}
                Err(e) => panic!("{op:?} failed: {e:?}"),
            }
        }
        Op::Modify { start, end, change } => {
            let calls = Cell::new(0);
            let failed_at = Cell::new(None);
            let result = mapping.modify_range(
                &window.region(start, end),
                &|chunk: &MemoryRegion, d: &mut UpdatableDescriptor<R::Attributes>| {
                    calls.set(calls.get() + 1);
                    if !d.flags().contains(R::Attributes::VALID) {
                        return Ok(());
                    }
                    let fail = match change {
                        Change::FailAt { page, .. } => window.page(chunk.start()) >= page,
                        Change::FailAfter { calls: limit, .. } => calls.get() > limit,
                        _ => false,
                    };
                    if fail {
                        failed_at.set(Some(window.page(chunk.start())));
                        return Err(());
                    }
                    match change {
                        Change::Flags { set, clear }
                        | Change::FailAt { set, clear, .. }
                        | Change::FailAfter { set, clear, .. } => d.modify_flags(set, clear),
                        Change::Unmap => {
                            d.modify_flags(R::Attributes::empty(), R::Attributes::VALID)
                        }
                        Change::Move { offset } => {
                            let flags = d.flags();
                            d.set(PhysicalAddress(d.output_address().0 + offset), flags)
                        }
                    }
                },
            );
            let apply = |page: Option<Page<R::Attributes>>| {
                let page = page?;
                match change {
                    Change::Flags { set, clear }
                    | Change::FailAt { set, clear, .. }
                    | Change::FailAfter { set, clear, .. } => Some(Page {
                        flags: (page.flags | set) & !clear,
                        ..page
                    }),
                    Change::Unmap => None,
                    Change::Move { offset } => Some(Page {
                        pa: page.pa + offset,
                        ..page
                    }),
                }
            };
            match result {
                Ok(()) => model.update(start, end, |_, page| apply(page)),
                // The updater failed while the change was being applied, after changing all the
                // valid descriptors before the one it failed at.
                Err(MapError::PteUpdateFault(_)) if failed_at.get().is_some() => {
                    model.update(start, failed_at.get().unwrap(), |_, page| apply(page));
                }
                // Updates of live mappings may be refused, which is also how failures of the
                // updater are reported if they occur during the check for break-before-make
                // violations.
                Err(MapError::BreakBeforeMakeViolation(_)) if live => {}
                Err(e) => panic!("{op:?} failed: {e:?}"),
            }
        }
        Op::Invalid(invalid) => {
            let (region, flags) = match invalid {
                Invalid::Backwards { start, end } => (
                    MemoryRegion::from(
                        VirtualAddress(window.va(end))..VirtualAddress(window.va(start)),
                    ),
                    R::Attributes::empty(),
                ),
                Invalid::OutOfRange => {
                    let size = mapping.size();
                    let edge = if window.base == 0 {
                        size
                    } else {
                        0usize.wrapping_sub(size)
                    };
                    (
                        MemoryRegion::new(edge - PAGE_SIZE, edge + PAGE_SIZE),
                        R::Attributes::empty(),
                    )
                }
                Invalid::TableOrPage { start, end, flags } => (window.region(start, end), flags),
            };
            let result =
                mapping.map_range(&region, PhysicalAddress(0), flags, Constraints::empty());
            let ok = match (invalid, &result) {
                (Invalid::Backwards { .. }, Err(MapError::RegionBackwards(_))) => true,
                (Invalid::OutOfRange, Err(MapError::AddressRange(_))) => true,
                (Invalid::TableOrPage { .. }, Err(MapError::InvalidFlags(_))) => true,
                // For live mappings, the check for break-before-make violations may fail first.
                (Invalid::TableOrPage { .. }, Err(MapError::BreakBeforeMakeViolation(_))) => live,
                _ => false,
            };
            assert!(ok, "{op:?} returned {result:?}");
            if !matches!(invalid, Invalid::TableOrPage { .. }) {
                let result = mapping.modify_range(&region, &|_, _| Ok(()));
                assert!(
                    result.is_err(),
                    "modify_range() for {op:?} returned {result:?}"
                );
            }
        }
        Op::Compact => mapping.compact_subtables(),
        Op::Activate => {
            // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
            // active for the sake of BBM rules and the TLB emulation.
            *ttbr = Some(unsafe { mapping.activate() });
        }
        Op::Deactivate => {
            // SAFETY: This doesn't actually deactivate the page table in tests, it just treats it
            // as inactive for the sake of BBM rules and the TLB emulation.
            unsafe { mapping.deactivate(ttbr.take().unwrap()) };
        }
    }
}

/// Applies a sequence of up to `ops` updates, chosen by `rng`, to a page table, and checks the page
/// table after each update. The updates are printed if `verbose` is set.
fn run<R: TranslationRegime>(variant: &Variant<R>, rng: &mut Rng<'_>, ops: usize, verbose: bool) {
    let mut mapping = (variant.new_mapping)();
    let mut model = Model::new(variant.window.pages);
    let mut ttbr = None;
    for i in 0..ops {
        if rng.exhausted() {
            break;
        }
        let mut op = variant.random_op(rng, ttbr.is_some());
        if ttbr.is_some() {
            op = op.shrink(|start, end| {
                count_descriptors(&mapping, &variant.window, start, end).1 <= LIVE_LIMIT
            });
            // For live mappings, modify_range() calls the updater for each descriptor to check for
            // break-before-make violations first, so make sure that the updater only fails while
            // the change is being applied.
            if let Op::Modify {
                start,
                end,
                change: Change::FailAfter { calls, .. },
            } = &mut op
            {
                let (all, _) = count_descriptors(&mapping, &variant.window, *start, *end);
                *calls = all + rng.below(all);
            }
        }
        if verbose {
            println!("{i}: {op:?}");
        }
        apply(&mut mapping, &mut model, &variant.window, &mut ttbr, op);
        check(&mapping, &model, &variant.window);
        // Check the TLB emulation for stale entries anywhere in the window, also after updates
        // that failed, rather than just in the updated range after successful updates.
        tlb::check_coherency(
            mapping.root_address(),
            &variant.window.region(0, variant.window.pages),
        );
    }
    if let Some(ttbr) = ttbr {
        // SAFETY: As above.
        unsafe { mapping.deactivate(ttbr) };
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
            // FNV-1a hash of the test name, so that each test uses different seeds.
            let base = name.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            });
            let runs = env_var("AARCH64_PAGING_FUZZ_RUNS").map_or(DEFAULT_RUNS, |n| n as usize);
            (0..runs as u64).map(|i| base.wrapping_add(i)).collect()
        }
    };
    for seed in seeds {
        println!("Seed {seed:#x}, {ops} operations");
        run(&variant, &mut Rng::SplitMix(seed), ops, true);
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

const EL1_MEMORY_TYPES: &[El1Attributes] = &[
    El1Attributes::ATTRIBUTE_INDEX_0,
    El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE),
    El1Attributes::ATTRIBUTE_INDEX_2.union(El1Attributes::OUTER_SHAREABLE),
];

/// Attributes that may be changed on live mappings, except for clearing `NON_GLOBAL`, and `USER`,
/// which require break-before-make. (`SWFLAG_2` is reserved for the TLB emulation.)
const EL1_OPTIONAL: &[El1Attributes] = &[
    El1Attributes::READ_ONLY,
    El1Attributes::ACCESSED,
    El1Attributes::PXN,
    El1Attributes::UXN,
    El1Attributes::DBM,
    El1Attributes::SWFLAG_0,
    El1Attributes::SWFLAG_3,
    El1Attributes::NON_GLOBAL,
    El1Attributes::USER,
];

fn el1_variant(
    new_mapping: fn() -> Mapping<IdTranslation<El1Attributes>, Emulated<El1And0>>,
    window: Window,
) -> Variant<Emulated<El1And0>> {
    Variant {
        new_mapping,
        window,
        base: El1Attributes::VALID.union(El1Attributes::ACCESSED),
        memory_types: EL1_MEMORY_TYPES,
        optional: EL1_OPTIONAL,
        software: El1Attributes::SWFLAG_1,
    }
}

fn el1and0() -> Variant<Emulated<El1And0>> {
    el1_variant(
        || {
            Mapping::with_asid_and_va_range(
                IdTranslation::new(),
                1,
                1,
                Emulated(El1And0),
                VaRange::Lower,
            )
        },
        LOWER_WINDOW,
    )
}

fn el1and0_upper() -> Variant<Emulated<El1And0>> {
    el1_variant(
        || {
            Mapping::with_asid_and_va_range(
                IdTranslation::new(),
                1,
                1,
                Emulated(El1And0),
                VaRange::Upper,
            )
        },
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
            Mapping::with_asid_and_va_range(
                IdTranslation::new(),
                2,
                1,
                Emulated(El2And0),
                VaRange::Lower,
            )
        },
        window: LOWER_WINDOW,
        base: El1Attributes::VALID
            .union(El1Attributes::ACCESSED)
            .union(El1Attributes::NON_GLOBAL),
        memory_types: EL1_MEMORY_TYPES,
        optional: EL2AND0_OPTIONAL,
        software: El1Attributes::SWFLAG_1,
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
        new_mapping: || Mapping::new(IdTranslation::new(), 1, Emulated(Stage2)),
        window: LOWER_WINDOW,
        base: Stage2Attributes::VALID.union(Stage2Attributes::ACCESS_FLAG),
        memory_types: STAGE2_MEMORY_TYPES,
        optional: STAGE2_OPTIONAL,
        software: Stage2Attributes::SWFLAG_1,
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
    let verbose = env::var_os("AARCH64_PAGING_FUZZ_VERBOSE").is_some();
    match variant % 4 {
        0 => run(&el1and0(), rng, MAX_FUZZ_OPS, verbose),
        1 => run(&el1and0_upper(), rng, MAX_FUZZ_OPS, verbose),
        2 => run(&el2and0_non_global(), rng, MAX_FUZZ_OPS, verbose),
        _ => run(&stage2(), rng, MAX_FUZZ_OPS, verbose),
    }
}

/// Applies the sequences of updates described by inputs provided by bolero, see the module
/// documentation.
#[test]
fn fuzz_inputs() {
    bolero::check!().for_each(fuzz_one);
}
