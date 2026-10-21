// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! A model of the TLB, used by the tests to check that live translation tables are only ever
//! updated in ways that cannot give rise to TLB conflicts or stale TLB entries.
//!
//! Tests opt in by using a translation regime wrapped in [`Emulated`], which makes the emulation
//! aware of the translation table hierarchies that are activated with it. The library itself only
//! reports the translation tables it allocates and frees, the stores to their descriptors, and the
//! barriers and TLB invalidation instructions it issues. While a hierarchy is live, the emulation
//! keeps track of the translation tables that are part of it, along with their levels and the
//! virtual addresses they translate, by following the stores to its table descriptors.
//!
//! The model is pessimistic: it assumes that the page table walker may fetch and cache any valid
//! descriptor of a live translation table at any time, and that a store to a descriptor is not
//! guaranteed to be observed by the walker until it has been completed by a DSB. More specifically:
//!
//! - Each store to a descriptor of a live translation table allocates a TLB entry for the new value
//!   of the descriptor if it is valid, as the walker may cache it right away.
//! - It also allocates a TLB entry for the old value of the descriptor if that was valid, as the
//!   walker may have cached it at any time before it was overwritten.
//! - Each such store marks the descriptor as pending, to record that the walker may still observe
//!   the old value. Descriptors stop being pending when a DSB is issued.
//! - When a table descriptor is overwritten, the walker may have cached any of the valid
//!   descriptors of the subtable it pointed to and of that subtable's own subtables, so TLB entries
//!   are allocated for all of them. Similarly, when a table descriptor is written, TLB entries are
//!   allocated for the valid descriptors of the subtable it points to and of its subtables, which
//!   the walker may now fetch.
//! - TLB invalidation (TLBI) marks the TLB entries it applies to for removal, and the removal takes
//!   effect once the TLBI is completed by a subsequent DSB that is not limited to stores. However, a
//!   TLBI does not affect TLB entries for which a descriptor at the same level or above on the walk
//!   path of the invalidated address is pending at the time the TLBI is issued: the walker may
//!   fetch the stale value again right after the invalidation, as the store that updated the
//!   descriptor may not be visible to it yet.
//! - TLBIs are modelled based on the operands that are passed to the actual instructions, taking
//!   the ASID and translation table level hint (TTL) fields into account, and TLBI by ASID does not
//!   affect global entries.
//! - Allocating a TLB entry that conflicts with an existing one, e.g., because it covers some of the
//!   same VA range but with a different output address, memory type or block size, or as a block
//!   mapping rather than a table, results in a panic. A TLB entry for a descriptor with the
//!   contiguous bit set is considered to cover the entire contiguous range, but as the contiguous
//!   bit is only a hint, a TLBI by address only applies to TLB entries for descriptors that
//!   actually map the address.
//!
//! In addition, after updating a live mapping, [`Mapping::map_range`](crate::Mapping::map_range)
//! and [`Mapping::modify_range`](crate::Mapping::modify_range) check that no stale TLB entries
//! remain for the updated range, as those functions guarantee that the updates have taken effect by
//! the time they return.
//!
//! Each thread has its own TLB, and TLB entries are tagged with the translation table hierarchy
//! they were created from, so that unrelated tests or mappings do not interfere with each other.

use crate::descriptor::{DescriptorBits, PhysicalAddress, VirtualAddress};
#[cfg(feature = "alloc")]
use crate::paging::VaRange;
use crate::paging::{
    BITS_PER_LEVEL, El1And0, El2And0, LEAF_LEVEL, MemoryRegion, PAGE_SIZE, Stage2,
    TranslationRegime, granularity_at_level,
};
use core::any::{Any, TypeId, type_name};
use core::cell::RefCell;
use core::fmt::{self, Display, Formatter};
use core::mem::size_of;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use std::string::String;
use std::vec::Vec;
use std::{format, thread_local};

const VALID: DescriptorBits = 1 << 0;
const TABLE_OR_PAGE: DescriptorBits = 1 << 1;
/// AttrIndx[2:0] and NS (stage 1) or MemAttr[3:0] (stage 2), and SH[1:0].
const MEMORY_ATTRIBUTES: DescriptorBits = (0b1111 << 2) | (0b11 << 8);
const NON_GLOBAL: DescriptorBits = 1 << 11;
const CONTIGUOUS: DescriptorBits = 1 << 52;
/// The bits reserved for software use, which are ignored by the walker.
const SOFTWARE: DescriptorBits = 0b1111 << 55;
/// PXNTable, UXNTable, APTable and NSTable.
const TABLE_ATTRIBUTES: DescriptorBits = 0b11111 << 59;
const OUTPUT_ADDRESS: DescriptorBits = ((1 << 48) - 1) & !(PAGE_SIZE - 1);
/// The number of descriptors covered by a single TLB entry if their contiguous bit is set.
const CONTIGUOUS_COUNT: usize = 16;
/// The number of descriptors in a translation table.
const ENTRIES: usize = 1 << BITS_PER_LEVEL;

/// The kind of DSB instruction, as far as its effect on the TLB is concerned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Dsb {
    /// A DSB that waits for all memory accesses and maintenance instructions to complete (e.g.
    /// `DSB ISH` or `DSB NSH`).
    Full,
    /// A DSB that only waits for stores to complete (e.g. `DSB ISHST`), and therefore does not
    /// complete TLB maintenance.
    Store,
}

/// A TLB invalidation instruction, along with its register operand where applicable.
#[derive(Clone, Copy, Debug)]
enum Tlbi {
    /// `TLBI VAAE1IS`: invalidates entries by VA, for all ASIDs.
    Vaae1is(usize),
    /// `TLBI VAE2IS` or `TLBI VAE3IS`: invalidates entries by VA. If the translation regime has
    /// ASIDs, non-global entries are only invalidated if they match the ASID in the operand.
    Vae(usize),
    /// `TLBI IPAS2E1IS`: invalidates stage 2 entries by IPA.
    Ipas2e1is(usize),
    /// `TLBI ASIDE1`: invalidates the non-global entries for the ASID in the operand.
    Aside1(usize),
    /// `TLBI VMALLE1IS`: invalidates all stage 1 entries of the EL1&0 translation regime (or of the
    /// EL2&0 translation regime if `HCR_EL2.{E2H, TGE}` is `{1, 1}`), but not stage 2 entries.
    Vmalle1is,
    /// `TLBI VMALLS12E1`: invalidates all stage 1 and stage 2 entries.
    Vmalls12e1,
}

impl Tlbi {
    /// Returns the address operand of the instruction, if it has one.
    fn address(self) -> Option<usize> {
        match self {
            // VA[55:12] in bits [43:0], and VA[63:56] are copies of VA[55].
            Self::Vaae1is(xt) | Self::Vae(xt) => Some((((xt << 20) as isize) >> 8) as usize),
            // IPA[51:12] in bits [39:0].
            Self::Ipas2e1is(xt) => Some((xt & ((1 << 40) - 1)) << 12),
            Self::Aside1(_) | Self::Vmalle1is | Self::Vmalls12e1 => None,
        }
    }

    /// Returns whether the instruction invalidates the given TLB entry.
    fn applies_to(self, entry: &Entry) -> bool {
        let maps_address = || self.address().is_some_and(|address| entry.maps(address));
        match self {
            Self::Vaae1is(xt) | Self::Ipas2e1is(xt) => maps_address() && ttl_matches(xt, entry),
            Self::Vae(xt) => {
                maps_address()
                    && ttl_matches(xt, entry)
                    && (entry.asid.is_none() || entry.is_global() || entry.asid == Some(xt >> 48))
            }
            Self::Aside1(xt) => !entry.is_global() && entry.asid == Some(xt >> 48),
            Self::Vmalle1is => {
                entry.regime == TypeId::of::<El1And0>() || entry.regime == TypeId::of::<El2And0>()
            }
            Self::Vmalls12e1 => true,
        }
    }
}

/// Returns whether the translation table level hint (TTL) in bits [47:44] of the operand `xt` of a
/// TLBI by address permits it to invalidate `entry`. If the hint is incorrect for the entry, the
/// architecture does not require the entry to be invalidated.
fn ttl_matches(xt: usize, entry: &Entry) -> bool {
    match ((xt >> 46) & 0b11, (xt >> 44) & 0b11) {
        // No hint.
        (0b00, _) => true,
        // The level of the leaf entry for the address, for the 4 KiB granule.
        (0b01, level @ 1..=3) if entry.is_table() => entry.level < level,
        (0b01, level @ 1..=3) => entry.level == level,
        // A reserved value, or the hint is for a different granule size.
        _ => false,
    }
}

/// Returns whether the walker treats `bits` as a valid descriptor at `level`.
fn is_valid(bits: DescriptorBits, level: usize) -> bool {
    match bits & (VALID | TABLE_OR_PAGE) {
        // A table or page descriptor.
        0b11 => true,
        // A block descriptor: these are not supported at level 0 without FEAT_LPA2.
        0b01 => level != 0 && level != LEAF_LEVEL,
        _ => false,
    }
}

/// A live translation table hierarchy, i.e., one that may be in use by the walker.
#[derive(Clone, Copy, Debug)]
struct Context {
    /// Identifies the translation table hierarchy, i.e., the address of its root table.
    id: usize,
    /// The level of the root table.
    level: usize,
    regime: TypeId,
    regime_name: &'static str,
    /// The ASID the translation table hierarchy is used with, if the regime has ASIDs.
    asid: Option<usize>,
}

/// Why a TLB entry was allocated.
#[derive(Clone, Copy, Debug)]
enum Origin {
    /// The descriptor was updated to this value, or it is part of a subtable that was connected to
    /// the hierarchy.
    Written,
    /// The descriptor held this value before it was updated.
    Overwritten,
    /// The descriptor is part of a subtable that was disconnected from the hierarchy.
    Disconnected,
}

/// A TLB entry, i.e., a cached copy of a valid descriptor.
#[derive(Clone, Debug)]
struct Entry {
    ctx: usize,
    regime: TypeId,
    regime_name: &'static str,
    asid: Option<usize>,
    /// The start of the VA range mapped by the descriptor the entry was created from.
    va: usize,
    /// The first and last VA covered by the entry, which differ from the VA range mapped by the
    /// descriptor if it is a block or page descriptor with its contiguous bit set.
    first: usize,
    last: usize,
    level: usize,
    bits: DescriptorBits,
    /// The address of the descriptor the entry was created from.
    desc: usize,
    origin: Origin,
    /// Set when the entry is invalidated by a TLBI, which takes effect at the next DSB.
    purge: bool,
}

impl Entry {
    fn is_table(&self) -> bool {
        self.level < LEAF_LEVEL && self.bits & TABLE_OR_PAGE != 0
    }

    /// Returns whether the entry applies to all ASIDs. Entries created from table descriptors are
    /// assumed to be tagged with the ASID used for the walk.
    fn is_global(&self) -> bool {
        !self.is_table() && self.bits & NON_GLOBAL == 0
    }

    /// Returns whether the descriptor the entry was created from maps `address`. As the contiguous
    /// bit is only a hint, the TLB may hold separate entries for each descriptor in a contiguous
    /// range, and so a TLBI by address is only guaranteed to invalidate entries for which this is
    /// the case.
    fn maps(&self, address: usize) -> bool {
        self.va <= address && address - self.va < granularity_at_level(self.level)
    }

    fn overlaps(&self, first: usize, last: usize) -> bool {
        self.first <= last && first <= self.last
    }

    /// Returns the output address that `self.first` translates to.
    fn base_output_address(&self) -> usize {
        (self.bits & OUTPUT_ADDRESS).wrapping_sub(self.va - self.first)
    }

    fn same_as(&self, other: &Self) -> bool {
        self.ctx == other.ctx
            && self.regime == other.regime
            && self.va == other.va
            && self.level == other.level
            && self.bits == other.bits
    }

    /// Returns why `self` cannot coexist in the TLB with `newer`, an entry allocated after it.
    fn conflict(&self, newer: &Self) -> Option<&'static str> {
        if self.ctx != newer.ctx
            || self.regime != newer.regime
            || !self.overlaps(newer.first, newer.last)
        {
            return None;
        }
        match (self.is_table(), newer.is_table()) {
            (true, true) => (self.level == newer.level
                && self.bits & OUTPUT_ADDRESS != newer.bits & OUTPUT_ADDRESS)
                .then_some("table descriptors pointing to different next-level tables"),
            (true, false) => (newer.level <= self.level).then_some("block versus table"),
            (false, true) => (self.level <= newer.level).then_some("block versus table"),
            (false, false) => {
                let diff = self.bits ^ newer.bits;
                if self.level != newer.level {
                    Some("different block sizes")
                } else if diff & CONTIGUOUS != 0 {
                    Some("different contiguous bits")
                } else if self.base_output_address() != newer.base_output_address() {
                    Some("different output addresses")
                } else if diff & MEMORY_ATTRIBUTES != 0 {
                    Some("different memory attributes")
                } else if self.asid.is_some() && !self.is_global() && newer.is_global() {
                    // The architecture permits changing a global mapping into a non-global one
                    // without break-before-make, but not the other way around.
                    Some("non-global mapping replaced by global mapping")
                } else if self.bits & CONTIGUOUS != 0 && diff & !(OUTPUT_ADDRESS | SOFTWARE) != 0 {
                    Some("live contiguous mapping modified")
                } else {
                    None
                }
            }
        }
    }

    /// Returns whether the entry is consistent with `current`, the value of the descriptor that
    /// the walker would fetch at the entry's level when translating the entry's VA, if the walk
    /// gets that far.
    fn is_consistent_with(&self, current: Option<DescriptorBits>) -> bool {
        let mask = if self.is_table() {
            OUTPUT_ADDRESS | TABLE_ATTRIBUTES | TABLE_OR_PAGE | VALID
        } else {
            !SOFTWARE
        };
        current.is_some_and(|bits| (bits ^ self.bits) & mask == 0)
    }
}

impl Display for Entry {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let kind = if self.is_table() {
            "table"
        } else if self.level == LEAF_LEVEL {
            "page"
        } else {
            "block"
        };
        write!(
            f,
            "{:#018x}..={:#018x}: level {} {} descriptor {:#018x} ({:?} value of the descriptor \
             at {:#x}, {}{})",
            self.first,
            self.last,
            self.level,
            kind,
            self.bits,
            self.origin,
            self.desc,
            self.regime_name,
            if self.purge {
                ", invalidation pending"
            } else {
                ""
            }
        )
    }
}

/// A descriptor that was updated without a DSB having been issued since, which means that the
/// walker may still observe its old value.
struct Pending {
    ctx: usize,
    /// The first and last VA of the range mapped by the descriptor.
    first: usize,
    last: usize,
    level: usize,
}

/// A translation table that is part of a live translation table hierarchy.
#[derive(Clone, Copy)]
struct Table {
    /// The root table of the hierarchy.
    ctx: usize,
    level: usize,
    /// The first VA translated by the table.
    base: usize,
}

/// A translation table allocated by the library.
#[derive(Clone, Copy)]
struct Allocation {
    /// The address of the table.
    table: usize,
    /// The level of the table, which is only needed if it is activated as a root table, i.e., by
    /// tests that use [`Emulated`].
    #[cfg_attr(not(feature = "alloc"), allow(dead_code))]
    level: usize,
}

#[derive(Default)]
struct Tlb {
    entries: Vec<Entry>,
    /// The pending descriptors, by address.
    pending: HashMap<usize, Pending>,
    /// The translation tables allocated by the library, by physical address.
    allocations: HashMap<usize, Allocation>,
    /// The physical addresses of the translation tables allocated by the library, by address.
    physical: HashMap<usize, usize>,
    /// The live translation table hierarchies, by the address of their root tables.
    hierarchies: HashMap<usize, Context>,
    /// The translation tables of the live translation table hierarchies, by address.
    tables: HashMap<usize, Table>,
    /// The translation table hierarchies that were activated with [`Emulated`] and not deactivated
    /// since, along with the regime and VA range they were activated for, in order of activation.
    #[cfg(feature = "alloc")]
    active: Vec<(usize, TypeId, VaRange)>,
}

thread_local! {
    static TLB: RefCell<Tlb> = RefCell::new(Tlb::default());
}

/// Runs `f` on the TLB of the current thread, unless it has been destroyed already.
fn with_tlb<T>(f: impl FnOnce(&mut Tlb) -> T) -> Option<T> {
    TLB.try_with(|tlb| f(&mut tlb.borrow_mut())).ok()
}

/// Panics with the message of `result` if it is an error.
fn report(result: Option<Result<(), String>>) {
    if let Some(Err(message)) = result {
        panic!("{message}");
    }
}

/// Returns the address, the current value and the first VA translated of each descriptor of the
/// translation table at address `table`, which translates the VA range starting at `base` at
/// `level`.
fn descriptors(
    table: usize,
    base: usize,
    level: usize,
) -> impl Iterator<Item = (usize, DescriptorBits, usize)> {
    let granularity = granularity_at_level(level);
    (0..ENTRIES).map(move |i| {
        let desc = table + i * size_of::<DescriptorBits>();
        (desc, load(desc), base.wrapping_add(i * granularity))
    })
}

/// Returns the current value of the descriptor at address `desc`.
fn load(desc: usize) -> DescriptorBits {
    // SAFETY: The emulation only accesses translation tables that were allocated by the library,
    // and the library reports when it frees them.
    unsafe { &*(desc as *const AtomicUsize) }.load(Ordering::Relaxed)
}

impl Tlb {
    /// Models the store of `new` to the descriptor at address `desc`, which held `old`, if the
    /// descriptor is part of a live translation table hierarchy.
    fn store(
        &mut self,
        desc: usize,
        old: DescriptorBits,
        new: DescriptorBits,
    ) -> Result<(), String> {
        let table = desc & !(PAGE_SIZE - 1);
        let Some(&Table { ctx, level, base }) = self.tables.get(&table) else {
            return Ok(());
        };
        let ctx = self.hierarchies[&ctx];
        let index = (desc - table) / size_of::<DescriptorBits>();
        let va = base.wrapping_add(index * granularity_at_level(level));
        self.store_at(&ctx, desc, old, new, va, level)?;
        let old_subtable = self.subtable(old, level);
        let new_subtable = self.subtable(new, level);
        if old_subtable != new_subtable {
            if let Some(subtable) = old_subtable {
                self.disconnect(&ctx, subtable, va, level + 1)?;
            }
            if let Some(subtable) = new_subtable {
                self.connect(&ctx, subtable, va, level + 1, Some(Origin::Written))?;
            }
        }
        Ok(())
    }

    /// Models the store of `new` to the descriptor at address `desc`, which held `old`, and which
    /// maps `va` at `level` in the live translation table hierarchy `ctx`.
    fn store_at(
        &mut self,
        ctx: &Context,
        desc: usize,
        old: DescriptorBits,
        new: DescriptorBits,
        va: usize,
        level: usize,
    ) -> Result<(), String> {
        let pending = self.pending.contains_key(&desc);
        if old == new && !pending {
            return Ok(());
        }
        // The walker may still observe the old value until the store is completed by a DSB.
        if !pending {
            let size = granularity_at_level(level);
            let first = va & !(size - 1);
            self.pending.insert(
                desc,
                Pending {
                    ctx: ctx.id,
                    first,
                    last: first + (size - 1),
                    level,
                },
            );
        }
        // The walker may have cached the old value at any time before the store, and it may cache
        // the new value at any time after it.
        self.allocate(ctx, old, va, level, desc, Origin::Overwritten)?;
        self.allocate(ctx, new, va, level, desc, Origin::Written)
    }

    /// Returns the address of the translation table that `bits` points to if it is a table
    /// descriptor at `level`, provided that the table was allocated by the library.
    fn subtable(&self, bits: DescriptorBits, level: usize) -> Option<usize> {
        if level == LEAF_LEVEL || bits & (VALID | TABLE_OR_PAGE) != VALID | TABLE_OR_PAGE {
            return None;
        }
        self.allocations
            .get(&(bits & OUTPUT_ADDRESS))
            .map(|allocation| allocation.table)
    }

    /// Makes the translation table at address `table`, which translates the VA range starting at
    /// `base` at `level`, part of the live translation table hierarchy `ctx`, along with its
    /// subtables. If `origin` is given, the walker may cache their valid descriptors from now on,
    /// and so TLB entries are allocated for them.
    fn connect(
        &mut self,
        ctx: &Context,
        table: usize,
        base: usize,
        level: usize,
        origin: Option<Origin>,
    ) -> Result<(), String> {
        self.tables.insert(
            table,
            Table {
                ctx: ctx.id,
                level,
                base,
            },
        );
        for (desc, bits, va) in descriptors(table, base, level) {
            if let Some(origin) = origin {
                self.allocate(ctx, bits, va, level, desc, origin)?;
            }
            if let Some(subtable) = self.subtable(bits, level) {
                self.connect(ctx, subtable, va, level + 1, origin)?;
            }
        }
        Ok(())
    }

    /// Removes the translation table at address `table`, which translates the VA range starting
    /// at `base` at `level`, and its subtables from the live translation table hierarchy `ctx`. As
    /// the walker may have cached any of their valid descriptors before, TLB entries are allocated
    /// for them.
    fn disconnect(
        &mut self,
        ctx: &Context,
        table: usize,
        base: usize,
        level: usize,
    ) -> Result<(), String> {
        self.tables.remove(&table);
        for (desc, bits, va) in descriptors(table, base, level) {
            self.allocate(ctx, bits, va, level, desc, Origin::Disconnected)?;
            if let Some(subtable) = self.subtable(bits, level) {
                self.disconnect(ctx, subtable, va, level + 1)?;
            }
        }
        Ok(())
    }

    fn allocate(
        &mut self,
        ctx: &Context,
        bits: DescriptorBits,
        va: usize,
        level: usize,
        desc: usize,
        origin: Origin,
    ) -> Result<(), String> {
        if !is_valid(bits, level) {
            return Ok(());
        }
        let size = granularity_at_level(level);
        let span = if bits & CONTIGUOUS != 0 && !(level < LEAF_LEVEL && bits & TABLE_OR_PAGE != 0) {
            size * CONTIGUOUS_COUNT
        } else {
            size
        };
        let first = va & !(span - 1);
        let entry = Entry {
            ctx: ctx.id,
            regime: ctx.regime,
            regime_name: ctx.regime_name,
            asid: ctx.asid,
            va: va & !(size - 1),
            first,
            last: first + (span - 1),
            level,
            bits,
            desc,
            origin,
            purge: false,
        };
        self.insert(entry)
    }

    fn insert(&mut self, entry: Entry) -> Result<(), String> {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.same_as(&entry)) {
            // The walker may cache the same value again after it has been invalidated.
            existing.purge = false;
            return Ok(());
        }
        if let Some((existing, reason)) = self
            .entries
            .iter()
            .find_map(|e| Some((e, e.conflict(&entry)?)))
        {
            return Err(format!(
                "TLB conflict ({reason}):\n  existing entry: {existing}\n  new entry:      {entry}"
            ));
        }
        self.entries.push(entry);
        Ok(())
    }

    fn tlbi(&mut self, regime: TypeId, op: Tlbi) {
        let purge = self
            .entries
            .iter()
            .map(|entry| {
                entry.regime == regime && op.applies_to(entry) && !self.may_refetch(op, entry)
            })
            .collect::<Vec<_>>();
        for (entry, purge) in self.entries.iter_mut().zip(purge) {
            entry.purge |= purge;
        }
    }

    /// Returns whether the walker may cache `entry` again right after `op` has invalidated it.
    fn may_refetch(&self, op: Tlbi, entry: &Entry) -> bool {
        // The walker may fetch the stale value again right after the invalidation if the update
        // of a descriptor at this level or above on the walk path of `address` is not visible to
        // it yet.
        let update_pending = |address: usize| {
            self.pending.values().any(|p| {
                p.ctx == entry.ctx
                    && p.level <= entry.level
                    && p.first <= address
                    && address <= p.last
            })
        };
        match op {
            // TLBI VMALLE1IS invalidates all addresses, including the one mapped by the entry.
            Tlbi::Vmalle1is => update_pending(entry.va),
            _ => op.address().is_some_and(update_pending),
        }
    }

    fn dsb(&mut self, kind: Dsb) {
        // All stores issued before the DSB are now visible to the walker.
        self.pending.clear();
        if kind == Dsb::Full {
            // All TLB maintenance issued before the DSB has completed.
            self.entries.retain(|entry| !entry.purge);
        }
    }

    /// Returns the descriptor at `level` that the walker would use to translate `va` in the live
    /// translation table hierarchy `ctx`, or `None` if the walk terminates at a higher level.
    fn lookup(&self, ctx: &Context, va: usize, level: usize) -> Option<DescriptorBits> {
        let mut table = ctx.id;
        let mut current = ctx.level;
        loop {
            let index = (va / granularity_at_level(current)) % ENTRIES;
            let bits = load(table + index * size_of::<DescriptorBits>());
            if current == level {
                return Some(bits);
            }
            table = self.subtable(bits, current)?;
            current += 1;
        }
    }

    /// Returns an error if the TLB holds entries for the translation table hierarchy `id` that
    /// overlap with `range` but that are inconsistent with the descriptors the walker would fetch
    /// now, as returned by `lookup(va, level)`.
    fn check_coherency(
        &self,
        id: usize,
        range: &MemoryRegion,
        lookup: impl Fn(usize, usize) -> Option<DescriptorBits>,
    ) -> Result<(), String> {
        if range.is_empty() {
            return Ok(());
        }
        let (first, last) = (range.start().0, range.end().0.wrapping_sub(1));
        for entry in self
            .entries
            .iter()
            .filter(|entry| entry.ctx == id && entry.overlaps(first, last))
        {
            let current = lookup(entry.va, entry.level);
            if !entry.is_consistent_with(current) {
                let current = match current {
                    Some(bits) => format!("{bits:#018x}"),
                    None => String::from("none (the walk terminates at a higher level)"),
                };
                return Err(format!(
                    "Stale TLB entry after TLB maintenance of {range}:\n  entry: {entry}\n  \
                     current descriptor: {current}"
                ));
            }
        }
        Ok(())
    }

    /// Models the translation table at address `table` being freed.
    fn freed(&mut self, table: usize) {
        self.pending
            .retain(|&desc, _| !(table..table + PAGE_SIZE).contains(&desc));
        // Forget all entries of a translation table hierarchy once its root table is freed.
        self.entries.retain(|entry| entry.ctx != table);
        self.tables.remove(&table);
    }
}

fn asid_of<R: TranslationRegime>(asid: R::Asid) -> Option<usize> {
    (&asid as &dyn Any).downcast_ref::<usize>().copied()
}

#[cfg(feature = "alloc")]
fn va_range_of<R: TranslationRegime>(va_range: R::VaRange) -> VaRange {
    (&va_range as &dyn Any)
        .downcast_ref::<VaRange>()
        .copied()
        .unwrap_or(VaRange::Lower)
}

/// Called after the library has allocated the translation table at address `table`, whose
/// physical address is `pa`, to be used at `level`.
pub(crate) fn table_allocated(table: usize, pa: PhysicalAddress, level: usize) {
    with_tlb(|tlb| {
        tlb.allocations.insert(pa.0, Allocation { table, level });
        tlb.physical.insert(table, pa.0);
    });
}

/// Called before the library deallocates the translation table at address `table`.
pub(crate) fn table_freed(table: usize) {
    with_tlb(|tlb| {
        if let Some(pa) = tlb.physical.remove(&table) {
            tlb.allocations.remove(&pa);
        }
        tlb.freed(table);
    });
}

/// Models the store of `new` to the descriptor `desc`, which held `old`.
pub(crate) fn store(desc: &AtomicUsize, old: DescriptorBits, new: DescriptorBits) {
    let desc = desc as *const AtomicUsize as usize;
    report(with_tlb(|tlb| tlb.store(desc, old, new)));
}

/// Models a DSB instruction.
pub(crate) fn dsb(kind: Dsb) {
    with_tlb(|tlb| tlb.dsb(kind));
}

/// Models the TLB invalidation instruction `TLBI <op>{, <Xt>}`, with register operand `xt` if it
/// takes one, issued for translation regime `R`.
pub(crate) fn tlbi<R: TranslationRegime>(op: &str, xt: Option<usize>) {
    let op = match (op, xt) {
        ("vaae1is", Some(xt)) => Tlbi::Vaae1is(xt),
        ("vae2is" | "vae3is", Some(xt)) => Tlbi::Vae(xt),
        ("ipas2e1is", Some(xt)) => Tlbi::Ipas2e1is(xt),
        ("aside1", Some(xt)) => Tlbi::Aside1(xt),
        ("vmalle1is", None) => Tlbi::Vmalle1is,
        ("vmalls12e1", None) => Tlbi::Vmalls12e1,
        _ => panic!("TLBI {op} is not supported by the TLB emulation"),
    };
    with_tlb(|tlb| tlb.tlbi(TypeId::of::<R>(), op));
}

/// Panics if the TLB holds entries that overlap with `range` for the live translation table
/// hierarchy whose root table is at physical address `root`, but that are inconsistent with the
/// descriptors the walker would fetch now.
pub(crate) fn check_coherency(root: PhysicalAddress, range: &MemoryRegion) {
    report(with_tlb(|tlb| {
        let Some(ctx) = tlb
            .allocations
            .get(&root.0)
            .and_then(|allocation| tlb.hierarchies.get(&allocation.table))
        else {
            return Ok(());
        };
        tlb.check_coherency(ctx.id, range, |va, level| tlb.lookup(ctx, va, level))
    }));
}

/// Models the TLB maintenance performed by [`TranslationRegime::deactivate`] for translation regime
/// `R`, after the translation table hierarchy that used `asid` has been deactivated.
fn invalidate_asid<R: TranslationRegime>(asid: R::Asid) {
    if let Some(asid) = asid_of::<R>(asid) {
        tlbi::<R>("aside1", Some(asid << 48));
    } else if TypeId::of::<R>() == TypeId::of::<Stage2>() {
        tlbi::<R>("vmalls12e1", None);
    }
    // EL2 and EL3 translation tables cannot be deactivated, so they are never invalidated.
    dsb(Dsb::Full);
}

/// A translation regime that behaves like `R`, but that also makes the TLB emulation aware of the
/// translation table hierarchies that are activated with it, so that their descriptors are subject
/// to its checks while they are live. The tests that build such hierarchies all depend on the
/// `alloc` feature.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Emulated<R>(pub(crate) R);

#[cfg(feature = "alloc")]
impl<R: TranslationRegime> TranslationRegime for Emulated<R> {
    type Attributes = R::Attributes;
    type Asid = R::Asid;
    type VaRange = R::VaRange;

    fn invalidate_va(va: VirtualAddress) {
        R::invalidate_va(va);
    }

    fn complete_tlb_maintenance() {
        R::complete_tlb_maintenance();
    }

    unsafe fn activate(root_pa: PhysicalAddress, asid: R::Asid, va_range: R::VaRange) -> usize {
        // SAFETY: Our caller promised to uphold the safety requirements of `activate`.
        let previous_ttbr = unsafe { R::activate(root_pa, asid, va_range) };
        let va_range = va_range_of::<R>(va_range);
        report(with_tlb(|tlb| {
            let Some(&Allocation { table, level }) = tlb.allocations.get(&root_pa.0) else {
                return Ok(());
            };
            tlb.active.push((table, TypeId::of::<R>(), va_range));
            if tlb.hierarchies.contains_key(&table) {
                return Ok(());
            }
            let ctx = Context {
                id: table,
                level,
                regime: TypeId::of::<R>(),
                regime_name: type_name::<R>(),
                asid: asid_of::<R>(asid),
            };
            let base = match va_range {
                VaRange::Lower => 0,
                // The upper range ends at the top of the address space, i.e. at 2^64.
                VaRange::Upper => (granularity_at_level(level) << BITS_PER_LEVEL).wrapping_neg(),
            };
            tlb.hierarchies.insert(table, ctx);
            tlb.connect(&ctx, table, base, level, None)
        }));
        previous_ttbr
    }

    unsafe fn deactivate(previous_ttbr: usize, asid: R::Asid, va_range: R::VaRange) {
        // SAFETY: Our caller promised to uphold the safety requirements of `deactivate`.
        unsafe { R::deactivate(previous_ttbr, asid, va_range) };
        invalidate_asid::<R>(asid);
        let key = (TypeId::of::<R>(), va_range_of::<R>(va_range));
        with_tlb(|tlb| {
            let Some(index) = tlb
                .active
                .iter()
                .rposition(|&(_, regime, va_range)| (regime, va_range) == key)
            else {
                return;
            };
            let (root, ..) = tlb.active.remove(index);
            if !tlb.active.iter().any(|&(table, ..)| table == root) {
                tlb.hierarchies.remove(&root);
                tlb.tables.retain(|_, table| table.ctx != root);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{Descriptor, El1Attributes, PagingAttributes};
    #[cfg(feature = "alloc")]
    use crate::idmap::IdMap;
    use crate::{wait_for_store_completion, wait_for_tlb_maintenance};
    use core::cell::Cell;

    /// Identifies the fictitious translation table hierarchy the test descriptors belong to.
    const ROOT: usize = 0x1_0000;
    const ASID: usize = 1;
    const BLOCK_SIZE: usize = PAGE_SIZE << 9;
    const VA: usize = 0x4000_0000;
    const PA: usize = 0x8000_0000;
    const NORMAL: El1Attributes = El1Attributes::VALID
        .union(El1Attributes::ATTRIBUTE_INDEX_1)
        .union(El1Attributes::INNER_SHAREABLE)
        .union(El1Attributes::ACCESSED);
    const DEVICE: El1Attributes = El1Attributes::VALID
        .union(El1Attributes::ATTRIBUTE_INDEX_0)
        .union(El1Attributes::ACCESSED);
    const PAGE: El1Attributes = NORMAL.union(El1Attributes::TABLE_OR_PAGE);
    const TABLE: El1Attributes = El1Attributes::VALID.union(El1Attributes::TABLE_OR_PAGE);
    const INVALID: El1Attributes = El1Attributes::empty();

    type Desc = Descriptor<El1Attributes>;

    thread_local! {
        /// The test translation table hierarchy, while it is live.
        static CONTEXT: Cell<Option<Context>> = const { Cell::new(None) };
    }

    /// Makes the test descriptors part of a fictitious translation table hierarchy until it is
    /// dropped. As they are not part of actual translation tables, the tests model the stores to
    /// them explicitly, along with the VA and level they map.
    struct TestHierarchy;

    impl Drop for TestHierarchy {
        fn drop(&mut self) {
            CONTEXT.with(|ctx| ctx.set(None));
        }
    }

    /// Starts accessing the test translation table hierarchy, used with `asid` in translation
    /// regime `R`. Stores to its descriptors are only modelled if it is `live`.
    fn enter_regime<R: TranslationRegime>(asid: R::Asid, live: bool) -> TestHierarchy {
        let ctx = Context {
            id: ROOT,
            level: 0,
            regime: TypeId::of::<R>(),
            regime_name: type_name::<R>(),
            asid: asid_of::<R>(asid),
        };
        CONTEXT.with(|current| current.set(live.then_some(ctx)));
        TestHierarchy
    }

    /// Starts accessing the test translation table hierarchy, in the EL1&0 translation regime.
    fn enter(live: bool) -> TestHierarchy {
        enter_regime::<El1And0>(ASID, live)
    }

    /// Stores a descriptor that maps `va` at `level`.
    fn set<A: PagingAttributes>(
        desc: &mut Descriptor<A>,
        va: usize,
        level: usize,
        pa: usize,
        flags: A,
    ) {
        let new = pa | flags.bits();
        let old = desc.0.swap(new, Ordering::Release);
        if let Some(ctx) = CONTEXT.with(Cell::get) {
            let desc = &desc.0 as *const AtomicUsize as usize;
            report(with_tlb(|tlb| {
                tlb.store_at(&ctx, desc, old, new, va, level)
            }));
        }
    }

    /// Invalidates `va` the way the library does for the EL1&0 translation regime.
    fn invalidate(va: usize) {
        El1And0::invalidate_va(VirtualAddress(va));
    }

    /// Returns whether the walker may not observe the last store to `desc` yet.
    fn is_pending<A: PagingAttributes>(desc: &Descriptor<A>) -> bool {
        let desc = &desc.0 as *const AtomicUsize as usize;
        with_tlb(|tlb| tlb.pending.contains_key(&desc)).unwrap()
    }

    /// Checks that the TLB holds no stale entries for `desc`, which maps `va` at `level`.
    fn check<A: PagingAttributes>(desc: &Descriptor<A>, va: usize, level: usize) {
        let size = granularity_at_level(level);
        report(with_tlb(|tlb| {
            tlb.check_coherency(ROOT, &MemoryRegion::new(va, va + size), |v, l| {
                (v == va && l == level).then(|| desc.bits())
            })
        }));
    }

    /// Returns the number of TLB entries for the test translation table hierarchy.
    fn entry_count() -> usize {
        with_tlb(|tlb| tlb.entries.iter().filter(|e| e.ctx == ROOT).count()).unwrap()
    }

    /// Creates a live level 2 block mapping at `VA`, and waits for it to become visible.
    fn live_block() -> Desc {
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 2, PA, NORMAL);
        wait_for_tlb_maintenance();
        desc
    }

    #[test]
    fn pending_tracks_store_completion() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        assert!(is_pending(&desc));
        wait_for_store_completion();
        assert!(!is_pending(&desc));

        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::READ_ONLY);
        assert!(is_pending(&desc));
        // Storing the same value again does not complete the previous store.
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::READ_ONLY);
        assert!(is_pending(&desc));
        wait_for_tlb_maintenance();
        assert!(!is_pending(&desc));
    }

    #[test]
    fn stores_allocate_entries() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        assert_eq!(entry_count(), 1);
        wait_for_store_completion();

        // Updating a valid descriptor allocates entries for its old and new values.
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::READ_ONLY);
        assert_eq!(entry_count(), 2);
        set(&mut desc, VA, 3, 0, INVALID);
        assert_eq!(entry_count(), 2);
        wait_for_store_completion();
        invalidate(VA);
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 0);

        // The old value of a live descriptor may be cached even if it was invalidated before.
        set(&mut desc, VA, 3, PA, PAGE);
        wait_for_store_completion();
        invalidate(VA);
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 0);
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::UXN);
        assert_eq!(entry_count(), 2);
    }

    #[test]
    fn inactive_tables_are_ignored() {
        let _ctx = enter(false);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 2, PA, NORMAL);
        set(&mut desc, VA, 2, PA + BLOCK_SIZE, NORMAL);
        set(&mut desc, VA, 2, 0x1234_5000, TABLE);
        assert!(!is_pending(&desc));
        assert_eq!(entry_count(), 0);
    }

    #[test]
    fn break_before_make() {
        let _ctx = enter(true);
        let mut desc = live_block();
        set(&mut desc, VA, 2, 0, INVALID);
        wait_for_store_completion();
        invalidate(VA);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 2, PA + BLOCK_SIZE, DEVICE);
        // Only the entry for the new value remains.
        assert_eq!(entry_count(), 1);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different output addresses)")]
    fn make_without_break() {
        let _ctx = enter(true);
        let mut desc = live_block();
        set(&mut desc, VA, 2, PA + BLOCK_SIZE, NORMAL);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different output addresses)")]
    fn break_before_make_without_tlbi() {
        let _ctx = enter(true);
        let mut desc = live_block();
        set(&mut desc, VA, 2, 0, INVALID);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 2, PA + BLOCK_SIZE, NORMAL);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different output addresses)")]
    fn break_before_make_without_dsb_before_tlbi() {
        let _ctx = enter(true);
        let mut desc = live_block();
        set(&mut desc, VA, 2, 0, INVALID);
        invalidate(VA);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 2, PA + BLOCK_SIZE, NORMAL);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different output addresses)")]
    fn break_before_make_without_dsb_after_tlbi() {
        let _ctx = enter(true);
        let mut desc = live_block();
        set(&mut desc, VA, 2, 0, INVALID);
        wait_for_store_completion();
        invalidate(VA);
        set(&mut desc, VA, 2, PA + BLOCK_SIZE, NORMAL);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different output addresses)")]
    fn break_before_make_with_store_dsb_after_tlbi() {
        let _ctx = enter(true);
        let mut desc = live_block();
        set(&mut desc, VA, 2, 0, INVALID);
        wait_for_store_completion();
        invalidate(VA);
        // A DSB that only waits for stores does not complete the TLBI.
        wait_for_store_completion();
        set(&mut desc, VA, 2, PA + BLOCK_SIZE, NORMAL);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different memory attributes)")]
    fn memory_type_change() {
        let _ctx = enter(true);
        let mut desc = live_block();
        set(&mut desc, VA, 2, PA, DEVICE);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different block sizes)")]
    fn page_overlapping_block() {
        let _ctx = enter(true);
        let _block = live_block();
        let mut page = Desc::EMPTY;
        set(&mut page, VA + PAGE_SIZE, 3, PA + PAGE_SIZE, PAGE);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (block versus table)")]
    fn table_replacing_block() {
        let _ctx = enter(true);
        let mut desc = live_block();
        set(&mut desc, VA, 2, 0x1234_5000, TABLE);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (block versus table)")]
    fn block_replacing_table() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 2, 0x1234_5000, TABLE);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 2, 0, INVALID);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 2, PA, NORMAL);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (table descriptors pointing to different next-level")]
    fn table_replacing_table() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 1, 0x1234_5000, TABLE);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 1, 0x1234_6000, TABLE);
    }

    #[test]
    fn permission_change() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        wait_for_tlb_maintenance();
        // Permissions may be changed without break-before-make, but they only take effect after
        // TLB maintenance.
        set(
            &mut desc,
            VA,
            3,
            PA,
            PAGE | El1Attributes::READ_ONLY | El1Attributes::UXN,
        );
        wait_for_store_completion();
        invalidate(VA);
        wait_for_tlb_maintenance();
        check(&desc, VA, 3);
    }

    #[test]
    #[should_panic(expected = "Stale TLB entry")]
    fn permission_change_without_tlbi() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::READ_ONLY);
        wait_for_tlb_maintenance();
        check(&desc, VA, 3);
    }

    #[test]
    #[should_panic(expected = "Stale TLB entry")]
    fn permission_change_without_dsb_before_tlbi() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::READ_ONLY);
        invalidate(VA);
        wait_for_tlb_maintenance();
        check(&desc, VA, 3);
    }

    #[test]
    fn tlbi_does_not_affect_entries_below_pending_update() {
        let _ctx = enter(true);
        let mut table = Desc::EMPTY;
        let mut page = Desc::EMPTY;
        set(&mut page, VA, 3, PA, PAGE);
        set(&mut table, VA, 2, 0x1234_5000, TABLE);
        wait_for_tlb_maintenance();
        set(&mut table, VA, 2, 0, INVALID);
        // The walker may fetch the stale table descriptor again, and the page descriptor through it.
        invalidate(VA);
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 2);
        // The invalidation takes effect once the update is visible to the walker.
        invalidate(VA);
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 0);
    }

    #[test]
    fn global_to_non_global() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::NON_GLOBAL);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (non-global mapping replaced by global mapping)")]
    fn non_global_to_global() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::NON_GLOBAL);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 3, PA, PAGE);
    }

    /// Maps a contiguous range of pages at `VA` to `pa`.
    fn map_contiguous(descs: &mut [Desc], pa: usize, flags: El1Attributes) {
        for (i, desc) in descs.iter_mut().enumerate() {
            let flags = flags | El1Attributes::CONTIGUOUS_OR_PROTECTED;
            set(desc, VA + i * PAGE_SIZE, 3, pa + i * PAGE_SIZE, flags);
        }
    }

    /// Remaps a live contiguous range of pages to a different output address using
    /// break-before-make, invalidating only the given number of pages.
    fn remap_contiguous(invalidated_pages: usize) {
        let _ctx = enter(true);
        let mut descs: [Desc; CONTIGUOUS_COUNT] = core::array::from_fn(|_| Desc::EMPTY);
        map_contiguous(&mut descs, PA, PAGE);
        wait_for_tlb_maintenance();
        for (i, desc) in descs.iter_mut().enumerate() {
            set(desc, VA + i * PAGE_SIZE, 3, 0, INVALID);
        }
        wait_for_store_completion();
        for i in 0..invalidated_pages {
            invalidate(VA + i * PAGE_SIZE);
        }
        wait_for_tlb_maintenance();
        map_contiguous(&mut descs, PA + BLOCK_SIZE, PAGE);
    }

    #[test]
    fn contiguous_break_before_make() {
        remap_contiguous(CONTIGUOUS_COUNT);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different output addresses)")]
    fn contiguous_break_before_make_with_single_tlbi() {
        remap_contiguous(1);
    }

    #[test]
    #[should_panic(expected = "TLB conflict (different contiguous bits)")]
    fn contiguous_bit_change() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        wait_for_tlb_maintenance();
        set(
            &mut desc,
            VA,
            3,
            PA,
            PAGE | El1Attributes::CONTIGUOUS_OR_PROTECTED,
        );
    }

    #[test]
    #[should_panic(expected = "TLB conflict (live contiguous mapping modified)")]
    fn contiguous_permission_change() {
        let _ctx = enter(true);
        let mut descs: [Desc; CONTIGUOUS_COUNT] = core::array::from_fn(|_| Desc::EMPTY);
        map_contiguous(&mut descs, PA, PAGE);
        wait_for_tlb_maintenance();
        let flags = PAGE | El1Attributes::CONTIGUOUS_OR_PROTECTED | El1Attributes::READ_ONLY;
        set(
            &mut descs[5],
            VA + 5 * PAGE_SIZE,
            3,
            PA + 5 * PAGE_SIZE,
            flags,
        );
    }

    /// Returns whether a TLBI with the given TTL hint invalidates the TLB entry for a live page.
    fn tlbi_with_ttl_invalidates_page(ttl: usize) -> bool {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        wait_for_tlb_maintenance();
        tlbi::<El1And0>("vaae1is", Some((VA >> 12) | (ttl << 44)));
        wait_for_tlb_maintenance();
        entry_count() == 0
    }

    #[test]
    fn tlbi_with_ttl_hint() {
        // 4 KiB granule, level 3.
        assert!(tlbi_with_ttl_invalidates_page(0b0111));
    }

    #[test]
    fn tlbi_with_wrong_level_hint() {
        // 4 KiB granule, level 2.
        assert!(!tlbi_with_ttl_invalidates_page(0b0110));
    }

    #[test]
    fn tlbi_with_wrong_granule_hint() {
        // 64 KiB granule, level 3.
        assert!(!tlbi_with_ttl_invalidates_page(0b1111));
    }

    #[test]
    fn tlbi_by_va_and_asid() {
        let _ctx = enter_regime::<El2And0>(ASID, true);
        let mut global = Desc::EMPTY;
        let mut non_global = Desc::EMPTY;
        set(&mut global, VA, 3, PA, PAGE);
        let flags = PAGE | El1Attributes::NON_GLOBAL;
        set(&mut non_global, VA + PAGE_SIZE, 3, PA + PAGE_SIZE, flags);
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 2);

        // Non-global entries for other ASIDs are not affected.
        tlbi::<El2And0>("vae2is", Some((VA >> 12) | ((ASID + 1) << 48)));
        tlbi::<El2And0>(
            "vae2is",
            Some(((VA + PAGE_SIZE) >> 12) | ((ASID + 1) << 48)),
        );
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 1);
        tlbi::<El2And0>("vae2is", Some(((VA + PAGE_SIZE) >> 12) | (ASID << 48)));
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 0);
    }

    #[test]
    fn deactivate_spares_global_entries() {
        let _ctx = enter(true);
        let mut global = Desc::EMPTY;
        let mut non_global = Desc::EMPTY;
        set(&mut global, VA, 3, PA, PAGE);
        let flags = PAGE | El1Attributes::NON_GLOBAL;
        set(&mut non_global, VA + PAGE_SIZE, 3, PA + PAGE_SIZE, flags);
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 2);
        invalidate_asid::<El1And0>(ASID);
        assert_eq!(entry_count(), 1);
    }

    /// Exercises updates of a live mapping through the public API, which must neither cause TLB
    /// conflicts nor leave stale TLB entries behind.
    #[cfg(feature = "alloc")]
    #[test]
    fn live_mapping() {
        const NORMAL_NG: El1Attributes = NORMAL.union(El1Attributes::NON_GLOBAL);
        let mut idmap = IdMap::with_asid(ASID, 1, Emulated(El1And0));
        idmap
            .map_range(&MemoryRegion::new(0, 2 * BLOCK_SIZE), NORMAL_NG)
            .unwrap();
        // SAFETY: This doesn't actually activate the page table in tests, it just treats it as
        // active for the sake of BBM rules and the TLB emulation.
        let ttbr = unsafe { idmap.activate() };

        // Map some pages, and change their permissions.
        let pages = MemoryRegion::new(2 * BLOCK_SIZE, 2 * BLOCK_SIZE + 8 * PAGE_SIZE);
        idmap.map_range(&pages, NORMAL_NG).unwrap();
        idmap
            .map_range(&pages, NORMAL_NG | El1Attributes::READ_ONLY)
            .unwrap();
        idmap
            .modify_range(&pages, &|_, d| {
                d.modify_flags(El1Attributes::UXN | El1Attributes::PXN, INVALID)
            })
            .unwrap();

        // Remap some of the pages with a different memory type, using break-before-make.
        let some = MemoryRegion::new(2 * BLOCK_SIZE + PAGE_SIZE, 2 * BLOCK_SIZE + 3 * PAGE_SIZE);
        idmap.map_range(&some, INVALID).unwrap();
        idmap.map_range(&some, DEVICE).unwrap();

        // Change the permissions of a block mapping, and unmap it.
        let block = MemoryRegion::new(BLOCK_SIZE, 2 * BLOCK_SIZE);
        idmap
            .map_range(&block, NORMAL_NG | El1Attributes::READ_ONLY)
            .unwrap();
        idmap
            .modify_range(&block, &|_, d| {
                d.modify_flags(INVALID, El1Attributes::VALID)
            })
            .unwrap();
        idmap.compact_subtables();

        // SAFETY: This doesn't actually deactivate the page table in tests, it just treats it as
        // inactive for the sake of BBM rules and the TLB emulation.
        unsafe { idmap.deactivate(ttbr) };

        // Make changes that require break-before-make while the mapping is inactive.
        idmap
            .map_range(&MemoryRegion::new(0, BLOCK_SIZE), DEVICE)
            .unwrap();
        idmap.map_range(&pages, NORMAL).unwrap();
        // SAFETY: As above.
        let ttbr = unsafe { idmap.activate() };
        idmap
            .map_range(
                &MemoryRegion::new(0, BLOCK_SIZE),
                DEVICE | El1Attributes::UXN,
            )
            .unwrap();
        // SAFETY: As above.
        unsafe { idmap.deactivate(ttbr) };
    }
}
