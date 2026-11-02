// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! The part of the randomized tests that doesn't depend on where they run: the generation of
//! random updates, the model of the expected mappings, and the checks of the page tables against
//! the model.
//!
//! This is shared with the randomized tests in `vmtests/`, which run in a virtual machine. So it
//! only uses the public API of the crate, through `crate::` paths that resolve to the modules of
//! the crate there as well, and `alloc` rather than `std`.

use crate::descriptor::{
    El1Attributes, PagingAttributes, PhysicalAddress, UpdatableDescriptor, VirtualAddress,
};
use crate::paging::{
    BITS_PER_LEVEL, Constraints, LEAF_LEVEL, MemoryRegion, PAGE_SIZE, TranslationRegime,
};
use crate::{MapError, Mapping};
use alloc::vec;
use alloc::vec::Vec;
use bitflags::Flags;
use core::cell::Cell;
use core::fmt;

/// The number of pages mapped by a level 2 block descriptor.
pub(super) const BLOCK_PAGES: usize = 1 << 9;
/// The number of pages mapped by a level 1 block descriptor.
pub(super) const HUGE_PAGES: usize = 1 << 18;
/// The size of the virtual address range that the tests operate on.
pub(super) const WINDOW_SIZE: usize = 4 << 30;

/// The page tables that the tests operate on, with the translation that the harness provides as
/// `TestTranslation`, as only the tests on the host can use `tlb::Tracked`.
pub(super) type TestMapping<R> =
    Mapping<super::TestTranslation<<R as TranslationRegime>::Attributes>, R>;

/// The source of the choices that determine a sequence of updates: either a small, deterministic
/// pseudo-random number generator (SplitMix64), or an input provided by bolero.
pub(super) enum Rng<'a> {
    SplitMix(u64),
    Input(&'a [u8]),
}

impl Rng<'_> {
    /// Returns a number in `0..n`.
    ///
    /// For an input, this consumes as few of its bytes as possible, so that a fuzzer that mutates a
    /// byte mostly affects a single choice, and it returns 0 once the input is used up.
    pub(super) fn below(&mut self, n: usize) -> usize {
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

/// Returns the seed of the first sequence of the test called `name`, so that each test uses
/// different seeds.
pub(super) fn first_seed(name: &str) -> u64 {
    // FNV-1a hash of the test name.
    name.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The range of virtual addresses that a test operates on, which is addressed in pages.
#[derive(Clone, Copy)]
pub(super) struct Window {
    pub(super) base: usize,
    pub(super) pages: usize,
}

impl Window {
    pub(super) fn va(&self, page: usize) -> usize {
        self.base.wrapping_add(page * PAGE_SIZE)
    }

    pub(super) fn page(&self, va: VirtualAddress) -> usize {
        va.0.wrapping_sub(self.base) / PAGE_SIZE
    }

    pub(super) fn region(&self, start: usize, end: usize) -> MemoryRegion {
        MemoryRegion::new(self.va(start), self.va(end))
    }
}

/// The range of physical addresses that mappings are created in.
#[derive(Clone, Copy)]
pub(super) struct PaSpace {
    pub(super) base: usize,
    pub(super) size: usize,
    /// Whether the range is so small that new mappings must be placed such that they fit in it,
    /// and that moving a mapping wraps around within it. Otherwise, new mappings are placed
    /// anywhere in the range, and moving them is assumed to never exceed the output address size.
    pub(super) wrap: bool,
}

impl PaSpace {
    /// Returns a random address in the range that is `offset` above a multiple of `align`, at
    /// which a mapping of `len` bytes may start.
    fn choose(&self, rng: &mut Rng<'_>, align: usize, offset: usize, len: usize) -> usize {
        let count = if self.wrap {
            assert!(offset + len <= self.size, "mapping doesn't fit");
            (self.size - offset - len) / align + 1
        } else {
            self.size / align
        };
        self.base + rng.below(count) * align + offset
    }

    /// Returns the physical address that an address in the model corresponds to, as the model
    /// doesn't wrap around when mappings are moved.
    pub(super) fn wrap(&self, pa: usize) -> usize {
        if self.wrap {
            self.base + (pa - self.base) % self.size
        } else {
            pa
        }
    }
}

/// What a page is expected to map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Page<A> {
    pub(super) pa: usize,
    pub(super) flags: A,
}

/// A run of consecutive pages that are either all unmapped, or all mapped with the same attributes
/// to consecutive physical pages. `page` describes the first page of the run.
#[derive(Clone, Copy, Debug)]
pub(super) struct Run<A> {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) page: Option<Page<A>>,
}

/// A model of what each page of the window is expected to map.
pub(super) struct Model<A> {
    pub(super) runs: Vec<Run<A>>,
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
pub(super) enum Change<A> {
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
pub(super) enum Invalid<A> {
    /// The range ends before it starts.
    Backwards { start: usize, end: usize },
    /// The range extends beyond the virtual address range of the page tables.
    OutOfRange,
    /// The attributes include `TABLE_OR_PAGE`, which `map_range` doesn't accept.
    TableOrPage { start: usize, end: usize, flags: A },
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Op<A> {
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
pub(super) struct Variant<R: TranslationRegime> {
    pub(super) new_mapping: fn() -> TestMapping<R>,
    pub(super) window: Window,
    /// The attributes that every valid mapping has.
    pub(super) base: R::Attributes,
    /// The memory types that mappings may have, including the shareability, which can only be
    /// changed using break-before-make.
    pub(super) memory_types: &'static [R::Attributes],
    /// Other attributes that are set or cleared at random, some of which can only be changed using
    /// break-before-make.
    pub(super) optional: &'static [R::Attributes],
    /// A software attribute that is sometimes set in invalid descriptors.
    pub(super) software: R::Attributes,
    /// The physical addresses that are mapped.
    pub(super) pa: PaSpace,
    /// The maximum number of valid descriptors that an update of a live page table may cover.
    pub(super) live_limit: usize,
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
                let len = (end - start) * PAGE_SIZE;
                let pa = if end - start > BLOCK_PAGES || rng.chance(80) {
                    let va = self.window.va(start);
                    let huge = HUGE_PAGES * PAGE_SIZE;
                    self.pa.choose(rng, huge, va % huge, len)
                } else {
                    self.pa.choose(rng, PAGE_SIZE, 0, len)
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

/// What runs the randomized test, which may add its own checks of the page tables.
pub(super) trait Harness<R: TranslationRegime> {
    /// Adds a line to the log of a sequence, which shows each update.
    fn log(&mut self, line: fmt::Arguments);

    /// Called before `op` is applied to the page tables.
    fn before(
        &mut self,
        _variant: &Variant<R>,
        _mapping: &TestMapping<R>,
        _model: &Model<R::Attributes>,
        _op: &Op<R::Attributes>,
    ) {
    }

    /// Called after `op` has been applied to the page tables, and the page tables have been
    /// checked against the model.
    fn after(
        &mut self,
        variant: &Variant<R>,
        mapping: &TestMapping<R>,
        model: &Model<R::Attributes>,
        op: &Op<R::Attributes>,
    );
}

/// Returns the size of the address range mapped by a descriptor at `level`.
fn granularity_at_level(level: usize) -> usize {
    PAGE_SIZE << ((LEAF_LEVEL - level) * BITS_PER_LEVEL)
}

/// Returns the number of block, page and invalid descriptors that map the pages in `start..end`,
/// and how many of those are valid.
fn count_descriptors<R: TranslationRegime>(
    mapping: &TestMapping<R>,
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
    mapping: &TestMapping<R>,
    model: &Model<R::Attributes>,
    variant: &Variant<R>,
) {
    let window = &variant.window;
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
                        pa: variant.pa.wrap(p.pa + (page - run.start) * PAGE_SIZE),
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
    mapping: &mut TestMapping<R>,
    model: &mut Model<R::Attributes>,
    variant: &Variant<R>,
    ttbr: &mut Option<usize>,
    op: Op<R::Attributes>,
) {
    let window = &variant.window;
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
                            let pa = variant.pa.wrap(d.output_address().0 + offset);
                            d.set(PhysicalAddress(pa), flags)
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
                    // The window is in the lower VA range if its base is.
                    let edge = if (window.base as isize) >= 0 {
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
            // SAFETY: In the host tests, this doesn't actually activate the page table, it just
            // treats it as active for the sake of BBM rules and the TLB emulation. The tests in a
            // virtual machine make sure that the page table maps the memory that they use in the
            // same way as the page table that it replaces.
            *ttbr = Some(unsafe { mapping.activate() });
        }
        Op::Deactivate => {
            // SAFETY: As above, and this switches back to the page table that it replaced.
            unsafe { mapping.deactivate(ttbr.take().unwrap()) };
        }
    }
}

/// Applies a sequence of up to `ops` updates, chosen by `rng`, to a page table, and checks the page
/// table after each update.
/// The page table that a sequence of updates is applied to, along with the value to restore the
/// TTBR to while it is active.
struct Subject<R: TranslationRegime> {
    mapping: TestMapping<R>,
    ttbr: Option<usize>,
}

/// Deactivates the page table if it is still active, e.g., because a check failed, as dropping an
/// active `Mapping` panics, which aborts the process while unwinding.
impl<R: TranslationRegime> Drop for Subject<R> {
    fn drop(&mut self) {
        if let Some(ttbr) = self.ttbr.take() {
            // SAFETY: As in `apply`, and this switches back to the page table that it replaced.
            unsafe { self.mapping.deactivate(ttbr) };
        }
    }
}

pub(super) fn run<R: TranslationRegime>(
    variant: &Variant<R>,
    rng: &mut Rng<'_>,
    ops: usize,
    harness: &mut impl Harness<R>,
) {
    let mut subject = Subject {
        mapping: (variant.new_mapping)(),
        ttbr: None,
    };
    let Subject { mapping, ttbr } = &mut subject;
    let mut model = Model::new(variant.window.pages);
    for i in 0..ops {
        if rng.exhausted() {
            break;
        }
        let mut op = variant.random_op(rng, ttbr.is_some());
        if ttbr.is_some() {
            op = op.shrink(|start, end| {
                count_descriptors(mapping, &variant.window, start, end).1 <= variant.live_limit
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
                let (all, _) = count_descriptors(mapping, &variant.window, *start, *end);
                *calls = all + rng.below(all);
            }
        }
        harness.log(format_args!("{i}: {op:?}"));
        harness.before(variant, mapping, &model, &op);
        apply(mapping, &mut model, variant, ttbr, op);
        check(mapping, &model, variant);
        harness.after(variant, mapping, &model, &op);
    }
}

pub(super) const EL1_MEMORY_TYPES: &[El1Attributes] = &[
    El1Attributes::ATTRIBUTE_INDEX_0,
    El1Attributes::ATTRIBUTE_INDEX_1.union(El1Attributes::INNER_SHAREABLE),
    El1Attributes::ATTRIBUTE_INDEX_2.union(El1Attributes::OUTER_SHAREABLE),
];

/// Attributes that may be changed on live mappings, except for clearing `NON_GLOBAL`, and `USER`,
/// which require break-before-make. (`SWFLAG_2` is reserved for the TLB emulation.)
pub(super) const EL1_OPTIONAL: &[El1Attributes] = &[
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
