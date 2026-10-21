// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! A model of the TLB, used by the tests to check that live translation tables are only ever
//! updated in ways that cannot give rise to TLB conflicts or stale TLB entries.
//!
//! The model is pessimistic: it assumes that the page table walker may fetch and cache any valid
//! descriptor of a live translation table at any time, and that a store to a descriptor is not
//! guaranteed to be observed by the walker until it has been completed by a DSB. More specifically:
//!
//! - Each store to a descriptor of a live translation table allocates a TLB entry for the new value
//!   of the descriptor if it is valid, as the walker may cache it right away.
//! - It also allocates a TLB entry for the old value of the descriptor if that was valid, as the
//!   walker may have cached it at any time before it was overwritten.
//! - Each such store sets the [`NEEDS_DSB`] software bit in the descriptor, to record that the
//!   walker may still observe the old value. The bit is cleared from all descriptors when a DSB is
//!   issued.
//! - TLB invalidation (TLBI) marks the TLB entries it applies to for removal, and the removal takes
//!   effect once the TLBI is completed by a subsequent DSB that is not limited to stores. However, a
//!   TLBI does not affect TLB entries for which a descriptor at the same level or above on the walk
//!   path of the invalidated address has its `NEEDS_DSB` bit set at the time the TLBI is issued:
//!   the walker may fetch the stale value again right after the invalidation, as the store that
//!   updated the descriptor may not be visible to it yet.
//! - TLBIs are modelled based on the operands that are passed to the actual instructions, taking
//!   the ASID and translation table level hint (TTL) fields into account, and TLBI by ASID does not
//!   affect global entries.
//! - Allocating a TLB entry that conflicts with an existing one, e.g., because it covers some of the
//!   same VA range but with a different output address, memory type or block size, or as a block
//!   mapping rather than a table, results in a panic. A TLB entry for a descriptor with the
//!   contiguous bit set is considered to cover the entire contiguous range, but as the contiguous
//!   bit is only a hint, a TLBI by address only applies to TLB entries for descriptors that
//!   actually map the address.
//! - Freeing a translation table while the TLB may still hold an entry for a table descriptor that
//!   points to it results in a panic, as the walker may still fetch descriptors from it.
//!
//! In addition, after updating a live mapping, [`Mapping::map_range`](crate::Mapping::map_range)
//! and [`Mapping::modify_range`](crate::Mapping::modify_range) check that no stale TLB entries
//! remain for the updated range, as those functions guarantee that the updates have taken effect by
//! the time they return.
//!
//! Each thread has its own TLB, and TLB entries are tagged with the translation table hierarchy
//! they were created from, so that unrelated tests or mappings do not interfere with each other.

use crate::descriptor::{DescriptorBits, PhysicalAddress, VirtualAddress};
use crate::paging::{
    LEAF_LEVEL, MemoryRegion, PAGE_SIZE, Stage2, TranslationRegime, granularity_at_level,
};
use core::any::{Any, TypeId, type_name};
use core::cell::RefCell;
use core::fmt::{self, Display, Formatter};
use core::sync::atomic::{AtomicUsize, Ordering};
use std::string::String;
use std::vec::Vec;
use std::{format, thread_local};

/// The software descriptor bit (`SWFLAG_2`) that the TLB emulation borrows to mark descriptors that
/// were updated without a DSB having been issued since, i.e., descriptors whose old value may still
/// be observed by the page table walker. It is hidden from `Descriptor::bits()`, and so it must not
/// be used by the tests.
pub(crate) const NEEDS_DSB: DescriptorBits = 1 << 57;

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
            Self::Aside1(_) | Self::Vmalls12e1 => None,
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

/// The translation table hierarchy that is being accessed.
#[derive(Clone, Copy, Debug)]
struct Context {
    /// Identifies the translation table hierarchy, i.e., the address of its root table.
    id: usize,
    regime: TypeId,
    regime_name: &'static str,
    /// The ASID the translation table hierarchy is used with, if the regime has ASIDs.
    asid: Option<usize>,
    /// Whether the translation table hierarchy may be in use by the walker.
    live: bool,
}

/// Why a TLB entry was allocated.
#[derive(Clone, Copy, Debug)]
enum Origin {
    /// The descriptor was updated to this value.
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

/// A descriptor whose `NEEDS_DSB` bit has been set.
struct Dirty {
    ctx: usize,
    desc: *const AtomicUsize,
    /// The first and last VA of the range mapped by the descriptor.
    first: usize,
    last: usize,
    level: usize,
}

impl Dirty {
    fn descriptor(&self) -> &AtomicUsize {
        // SAFETY: Descriptors are removed from the list of dirty descriptors before the translation
        // tables containing them are deallocated.
        unsafe { &*self.desc }
    }
}

#[derive(Default)]
struct Tlb {
    /// The translation table hierarchy that is currently being accessed, if any.
    context: Option<Context>,
    entries: Vec<Entry>,
    /// The descriptors that currently have their `NEEDS_DSB` bit set.
    dirty: Vec<Dirty>,
}

thread_local! {
    static TLB: RefCell<Tlb> = RefCell::new(Tlb::default());
}

/// Runs `f` on the TLB of the current thread, unless it has been destroyed already.
fn with_tlb<T>(f: impl FnOnce(&mut Tlb) -> T) -> Option<T> {
    TLB.try_with(|tlb| f(&mut tlb.borrow_mut())).ok()
}

impl Tlb {
    fn store(
        &mut self,
        desc: &AtomicUsize,
        old: DescriptorBits,
        new: DescriptorBits,
        va: usize,
        level: usize,
    ) -> Result<(), String> {
        let Some(ctx) = self.context.filter(|ctx| ctx.live) else {
            return Ok(());
        };
        let pending = old & NEEDS_DSB != 0;
        let old = old & !NEEDS_DSB;
        if old == new && !pending {
            return Ok(());
        }
        // The walker may still observe the old value until the store is completed by a DSB.
        desc.fetch_or(NEEDS_DSB, Ordering::Relaxed);
        if !pending {
            let size = granularity_at_level(level);
            let first = va & !(size - 1);
            self.dirty.push(Dirty {
                ctx: ctx.id,
                desc,
                first,
                last: first + (size - 1),
                level,
            });
        }
        // The walker may have cached the old value at any time before the store, and it may cache
        // the new value at any time after it.
        let address = desc as *const AtomicUsize as usize;
        self.allocate(&ctx, old, va, level, address, Origin::Overwritten)?;
        self.allocate(&ctx, new, va, level, address, Origin::Written)
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
        let address = op.address();
        let dirty = &self.dirty;
        for entry in self
            .entries
            .iter_mut()
            .filter(|entry| entry.regime == regime && op.applies_to(entry))
        {
            // The walker may fetch the stale value again right after the invalidation if the update
            // of a descriptor at this level or above is not visible to it yet.
            let refetch = address.is_some_and(|address| {
                dirty.iter().any(|d| {
                    d.ctx == entry.ctx
                        && d.level <= entry.level
                        && d.first <= address
                        && address <= d.last
                        && d.descriptor().load(Ordering::Relaxed) & NEEDS_DSB != 0
                })
            });
            if !refetch {
                entry.purge = true;
            }
        }
    }

    fn dsb(&mut self, kind: Dsb) {
        // All stores issued before the DSB are now visible to the walker.
        for dirty in self.dirty.drain(..) {
            dirty.descriptor().fetch_and(!NEEDS_DSB, Ordering::Relaxed);
        }
        if kind == Dsb::Full {
            // All TLB maintenance issued before the DSB has completed.
            self.entries.retain(|entry| !entry.purge);
        }
    }
}

/// Makes the TLB emulation aware of the translation table hierarchy that is being accessed, until
/// it is dropped.
pub(crate) struct ContextGuard(Option<Context>);

impl ContextGuard {
    /// Starts accessing the translation table hierarchy with root table `id`, used with `asid` in
    /// translation regime `R`. Stores to its descriptors are only modelled if it is `live`.
    pub(crate) fn enter<R: TranslationRegime>(id: usize, asid: R::Asid, live: bool) -> Self {
        let context = Context {
            id,
            regime: TypeId::of::<R>(),
            regime_name: type_name::<R>(),
            asid: asid_of::<R>(asid),
            live,
        };
        Self(with_tlb(|tlb| tlb.context.replace(context)).flatten())
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        let previous = self.0.take();
        with_tlb(|tlb| tlb.context = previous);
    }
}

fn asid_of<R: TranslationRegime>(asid: R::Asid) -> Option<usize> {
    (&asid as &dyn Any).downcast_ref::<usize>().copied()
}

/// Checks that `bits` does not use the software bit that is reserved for the TLB emulation.
pub(crate) fn check_reserved_bits(bits: DescriptorBits) {
    assert_eq!(
        bits & NEEDS_DSB,
        0,
        "Descriptor bit {NEEDS_DSB:#x} is reserved for the TLB emulation in tests"
    );
}

/// Models the store of `new` to the descriptor `desc` which held `old`, and which maps `va` at
/// `level` in the current translation table hierarchy.
pub(crate) fn store(
    desc: &AtomicUsize,
    old: DescriptorBits,
    new: DescriptorBits,
    va: VirtualAddress,
    level: usize,
) {
    check_reserved_bits(new);
    if let Some(Err(conflict)) = with_tlb(|tlb| tlb.store(desc, old, new, va.0, level)) {
        panic!("{conflict}");
    }
}

/// Models the walker having cached the descriptor at `desc` with value `bits`, which maps `va` at
/// `level` in a subtable that is about to be disconnected from the current translation table
/// hierarchy.
pub(crate) fn disconnect(bits: DescriptorBits, va: VirtualAddress, level: usize, desc: usize) {
    let result = with_tlb(|tlb| {
        let Some(ctx) = tlb.context.filter(|ctx| ctx.live) else {
            return Ok(());
        };
        tlb.allocate(&ctx, bits, va.0, level, desc, Origin::Disconnected)
    });
    if let Some(Err(conflict)) = result {
        panic!("{conflict}");
    }
}

/// Models a DSB instruction.
pub(crate) fn dsb(kind: Dsb) {
    with_tlb(|tlb| tlb.dsb(kind));
}

/// Models the TLB invalidation instruction `TLBI <op>, <Xt>` with register operand `xt`, issued for
/// translation regime `R`.
pub(crate) fn tlbi<R: TranslationRegime>(op: &str, xt: usize) {
    let op = match op {
        "vaae1is" => Tlbi::Vaae1is(xt),
        "vae2is" | "vae3is" => Tlbi::Vae(xt),
        "ipas2e1is" => Tlbi::Ipas2e1is(xt),
        "aside1" => Tlbi::Aside1(xt),
        _ => panic!("TLBI {op} is not supported by the TLB emulation"),
    };
    invalidate::<R>(op);
}

/// Models TLB invalidation for translation regime `R`.
fn invalidate<R: TranslationRegime>(op: Tlbi) {
    with_tlb(|tlb| tlb.tlbi(TypeId::of::<R>(), op));
}

/// Models the TLB maintenance performed by [`TranslationRegime::deactivate`] for translation regime
/// `R`, after the translation table hierarchy that used `asid` has been deactivated.
pub(crate) fn deactivate<R: TranslationRegime>(asid: R::Asid) {
    if let Some(asid) = asid_of::<R>(asid) {
        tlbi::<R>("aside1", asid << 48);
    } else if TypeId::of::<R>() == TypeId::of::<Stage2>() {
        invalidate::<R>(Tlbi::Vmalls12e1);
    }
    // EL2 and EL3 translation tables cannot be deactivated, so they are never invalidated.
    dsb(Dsb::Full);
}

/// Called before the translation table at address `table`, whose physical address is `pa`, is
/// deallocated.
pub(crate) fn table_freed(table: usize, pa: PhysicalAddress) {
    let result = with_tlb(|tlb| {
        tlb.dirty
            .retain(|d| !(table..table + PAGE_SIZE).contains(&(d.desc as usize)));
        // Forget all entries of a translation table hierarchy once its root table is freed.
        tlb.entries.retain(|entry| entry.ctx != table);
        // The walker may still fetch descriptors from the table via a cached table descriptor that
        // points to it, so the table must not be freed before such entries have been invalidated.
        // Don't report this while unwinding, e.g., from a panic raised by the TLB emulation while
        // the table hierarchy was being updated, as panicking again would abort the test run.
        match tlb
            .entries
            .iter()
            .find(|entry| entry.is_table() && entry.bits & OUTPUT_ADDRESS == pa.0)
        {
            Some(entry) if !std::thread::panicking() => Err(format!(
                "Translation table at {pa} freed while the TLB may still refer to it:\n  {entry}"
            )),
            _ => Ok(()),
        }
    });
    if let Some(Err(message)) = result {
        panic!("{message}");
    }
}

/// Panics if the TLB holds entries for the translation table hierarchy with root table `id` that
/// overlap with `range` but that are inconsistent with the descriptors the walker would fetch now,
/// as returned by `lookup(va, level)`.
pub(crate) fn check_coherency(
    id: usize,
    range: &MemoryRegion,
    lookup: impl Fn(VirtualAddress, usize) -> Option<DescriptorBits>,
) {
    if range.is_empty() {
        return;
    }
    let (first, last) = (range.start().0, range.end().0.wrapping_sub(1));
    let entries = with_tlb(|tlb| {
        tlb.entries
            .iter()
            .filter(|entry| entry.ctx == id && entry.overlaps(first, last))
            .cloned()
            .collect::<Vec<_>>()
    })
    .unwrap_or_default();
    for entry in entries {
        let current = lookup(VirtualAddress(entry.va), entry.level);
        if !entry.is_consistent_with(current) {
            let current = match current {
                Some(bits) => format!("{bits:#018x}"),
                None => String::from("none (the walk terminates at a higher level)"),
            };
            panic!(
                "Stale TLB entry after TLB maintenance of {range}:\n  entry: {entry}\n  \
                 current descriptor: {current}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{Descriptor, El1Attributes, PhysicalAddress};
    #[cfg(feature = "alloc")]
    use crate::idmap::IdMap;
    use crate::paging::{El1And0, El2And0};
    use crate::{wait_for_store_completion, wait_for_tlb_maintenance};

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

    /// Starts accessing the test translation table hierarchy, in the EL1&0 translation regime.
    fn enter(live: bool) -> ContextGuard {
        ContextGuard::enter::<El1And0>(ROOT, ASID, live)
    }

    /// Stores a descriptor that maps `va` at `level`.
    fn set(desc: &mut Desc, va: usize, level: usize, pa: usize, flags: El1Attributes) {
        desc.set_at(PhysicalAddress(pa), flags, VirtualAddress(va), level);
    }

    /// Invalidates `va` the way the library does for the EL1&0 translation regime.
    fn invalidate(va: usize) {
        El1And0::invalidate_va(VirtualAddress(va));
    }

    fn raw_bits(desc: &Desc) -> DescriptorBits {
        desc.0.load(Ordering::Relaxed)
    }

    /// Checks that the TLB holds no stale entries for `desc`, which maps `va` at `level`.
    fn check(desc: &Desc, va: usize, level: usize) {
        let size = granularity_at_level(level);
        check_coherency(ROOT, &MemoryRegion::new(va, va + size), |v, l| {
            (v.0 == va && l == level).then(|| desc.bits())
        });
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
    fn needs_dsb_tracks_store_completion() {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE);
        // The bit is set in memory, but hidden from users of the descriptor.
        assert_eq!(raw_bits(&desc), PA | PAGE.bits() | NEEDS_DSB);
        assert_eq!(desc.bits(), PA | PAGE.bits());
        wait_for_store_completion();
        assert_eq!(raw_bits(&desc), PA | PAGE.bits());

        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::READ_ONLY);
        assert_ne!(raw_bits(&desc) & NEEDS_DSB, 0);
        // Storing the same value again does not complete the previous store.
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::READ_ONLY);
        assert_ne!(raw_bits(&desc) & NEEDS_DSB, 0);
        wait_for_tlb_maintenance();
        assert_eq!(raw_bits(&desc) & NEEDS_DSB, 0);
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
        assert_eq!(raw_bits(&desc) & NEEDS_DSB, 0);
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

    /// Unmaps a live table descriptor, and frees the table it pointed to after performing the given
    /// TLB maintenance.
    fn free_unmapped_table(maintenance: impl FnOnce()) {
        let _ctx = enter(true);
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 2, 0x1234_5000, TABLE);
        wait_for_tlb_maintenance();
        set(&mut desc, VA, 2, 0, INVALID);
        maintenance();
        table_freed(0x2_0000, PhysicalAddress(0x1234_5000));
    }

    #[test]
    fn free_table_after_tlb_maintenance() {
        free_unmapped_table(|| {
            wait_for_store_completion();
            invalidate(VA);
            wait_for_tlb_maintenance();
        });
    }

    #[test]
    #[should_panic(expected = "freed while the TLB may still refer to it")]
    fn free_table_without_tlbi() {
        free_unmapped_table(wait_for_tlb_maintenance);
    }

    #[test]
    #[should_panic(expected = "freed while the TLB may still refer to it")]
    fn free_table_before_tlbi_completes() {
        free_unmapped_table(|| {
            wait_for_store_completion();
            invalidate(VA);
        });
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
        tlbi::<El1And0>("vaae1is", (VA >> 12) | (ttl << 44));
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
        let _ctx = ContextGuard::enter::<El2And0>(ROOT, ASID, true);
        let mut global = Desc::EMPTY;
        let mut non_global = Desc::EMPTY;
        set(&mut global, VA, 3, PA, PAGE);
        let flags = PAGE | El1Attributes::NON_GLOBAL;
        set(&mut non_global, VA + PAGE_SIZE, 3, PA + PAGE_SIZE, flags);
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 2);

        // Non-global entries for other ASIDs are not affected.
        tlbi::<El2And0>("vae2is", (VA >> 12) | ((ASID + 1) << 48));
        tlbi::<El2And0>("vae2is", ((VA + PAGE_SIZE) >> 12) | ((ASID + 1) << 48));
        wait_for_tlb_maintenance();
        assert_eq!(entry_count(), 1);
        tlbi::<El2And0>("vae2is", ((VA + PAGE_SIZE) >> 12) | (ASID << 48));
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
        deactivate::<El1And0>(ASID);
        assert_eq!(entry_count(), 1);
    }

    #[test]
    #[should_panic(expected = "reserved for the TLB emulation")]
    fn reserved_bit() {
        let mut desc = Desc::EMPTY;
        set(&mut desc, VA, 3, PA, PAGE | El1Attributes::SWFLAG_2);
    }

    /// Exercises updates of a live mapping through the public API, which must neither cause TLB
    /// conflicts nor leave stale TLB entries behind.
    #[cfg(feature = "alloc")]
    #[test]
    fn live_mapping() {
        const NORMAL_NG: El1Attributes = NORMAL.union(El1Attributes::NON_GLOBAL);
        let mut idmap = IdMap::with_asid(ASID, 1, El1And0);
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
