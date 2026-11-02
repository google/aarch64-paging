// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Exception handling, and memory accesses that report faults rather than causing a panic.

use core::arch::{asm, global_asm};
use core::fmt::{self, Display, Formatter};

global_asm!(
    r#"
    .macro  unexpected offset
    .balign 0x80
    mov     x0, #\offset
    b       {unexpected}
    .endm

    .section .text.vectors, "ax"
    .balign 0x800
    .global vector_table
vector_table:
    unexpected 0x000
    unexpected 0x080
    unexpected 0x100
    unexpected 0x180
    /* Synchronous exception from the current EL, using SP_ELx. */
    .balign 0x80
    b       .Lsync
    unexpected 0x280
    unexpected 0x300
    unexpected 0x380
    unexpected 0x400
    unexpected 0x480
    unexpected 0x500
    unexpected 0x580
    unexpected 0x600
    unexpected 0x680
    unexpected 0x700
    unexpected 0x780

.Lsync:
    sub     sp, sp, #(8 * 34)
    stp     x0, x1, [sp, #(8 * 0)]
    stp     x2, x3, [sp, #(8 * 2)]
    stp     x4, x5, [sp, #(8 * 4)]
    stp     x6, x7, [sp, #(8 * 6)]
    stp     x8, x9, [sp, #(8 * 8)]
    stp     x10, x11, [sp, #(8 * 10)]
    stp     x12, x13, [sp, #(8 * 12)]
    stp     x14, x15, [sp, #(8 * 14)]
    stp     x16, x17, [sp, #(8 * 16)]
    stp     x18, x19, [sp, #(8 * 18)]
    stp     x20, x21, [sp, #(8 * 20)]
    stp     x22, x23, [sp, #(8 * 22)]
    stp     x24, x25, [sp, #(8 * 24)]
    stp     x26, x27, [sp, #(8 * 26)]
    stp     x28, x29, [sp, #(8 * 28)]
    mrs     x0, elr_el1
    stp     x30, x0, [sp, #(8 * 30)]
    mrs     x0, spsr_el1
    str     x0, [sp, #(8 * 32)]
    mov     x0, sp
    bl      {sync}
    ldp     x30, x0, [sp, #(8 * 30)]
    msr     elr_el1, x0
    ldr     x0, [sp, #(8 * 32)]
    msr     spsr_el1, x0
    ldp     x0, x1, [sp, #(8 * 0)]
    ldp     x2, x3, [sp, #(8 * 2)]
    ldp     x4, x5, [sp, #(8 * 4)]
    ldp     x6, x7, [sp, #(8 * 6)]
    ldp     x8, x9, [sp, #(8 * 8)]
    ldp     x10, x11, [sp, #(8 * 10)]
    ldp     x12, x13, [sp, #(8 * 12)]
    ldp     x14, x15, [sp, #(8 * 14)]
    ldp     x16, x17, [sp, #(8 * 16)]
    ldp     x18, x19, [sp, #(8 * 18)]
    ldp     x20, x21, [sp, #(8 * 20)]
    ldp     x22, x23, [sp, #(8 * 22)]
    ldp     x24, x25, [sp, #(8 * 24)]
    ldp     x26, x27, [sp, #(8 * 26)]
    ldp     x28, x29, [sp, #(8 * 28)]
    add     sp, sp, #(8 * 34)
    eret

    /*
     * Loads from the address in x0, and returns the value in x0, and 0 in x1, or the ESR with bit
     * 63 set in x1 if the load causes a data abort.
     */
    .section .text.probe, "ax"
    .global probe_read
probe_read:
    mov     x1, xzr
.Lprobe_read_access:
    ldr     x0, [x0]
    ret

    /*
     * Stores x1 to the address in x0, and returns 0 in x0, or the ESR with bit 63 set if the store
     * causes a data abort.
     */
    .global probe_write
probe_write:
    mov     x2, xzr
.Lprobe_write_access:
    str     x1, [x0]
    mov     x0, x2
    ret

    .section .rodata.probe, "a"
    .balign 8
    .global probe_read_access
probe_read_access:
    .quad   .Lprobe_read_access
    .global probe_write_access
probe_write_access:
    .quad   .Lprobe_write_access
"#,
    unexpected = sym unexpected,
    sync = sym sync,
);

/// The registers that are saved on exception entry.
#[repr(C)]
struct Frame {
    x: [u64; 31],
    elr: u64,
    spsr: u64,
    _padding: u64,
}

/// Set in the ESR values that are returned for faulting accesses, so that they're never 0.
const FAULTED: u64 = 1 << 63;
/// The exception class of a data abort taken without a change in exception level.
const EC_DATA_ABORT_SAME_EL: u64 = 0x25;

unsafe extern "C" {
    fn probe_read(va: usize) -> ReadResult;
    fn probe_write(va: usize, value: u64) -> u64;
    /// The addresses of the instructions in `probe_read` and `probe_write` that access memory.
    static probe_read_access: u64;
    static probe_write_access: u64;
}

#[repr(C)]
struct ReadResult {
    value: u64,
    esr: u64,
}

fn esr() -> u64 {
    let esr: u64;
    // SAFETY: Reading ESR has no side effects.
    unsafe { asm!("mrs {}, esr_el1", out(reg) esr, options(nomem, nostack)) };
    esr
}

fn far() -> u64 {
    let far: u64;
    // SAFETY: Reading FAR has no side effects.
    unsafe { asm!("mrs {}, far_el1", out(reg) far, options(nomem, nostack)) };
    far
}

extern "C" fn sync(frame: &mut Frame) {
    let esr = esr();
    if (esr >> 26) & 0x3f == EC_DATA_ABORT_SAME_EL {
        // SAFETY: These are initialised, and never written.
        let (read, write) = unsafe { (probe_read_access, probe_write_access) };
        // Return the ESR from the probe function, skipping the access.
        if frame.elr == read {
            frame.x[1] = esr | FAULTED;
            frame.elr += 4;
            return;
        } else if frame.elr == write {
            frame.x[2] = esr | FAULTED;
            frame.elr += 4;
            return;
        }
    }
    panic!(
        "Unexpected synchronous exception: ESR {esr:#x}, FAR {:#x}, ELR {:#x}",
        far(),
        frame.elr
    );
}

extern "C" fn unexpected(offset: u64) -> ! {
    panic!(
        "Unexpected exception at vector offset {offset:#x}: ESR {:#x}, FAR {:#x}",
        esr(),
        far()
    );
}

/// The outcome of a memory access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fault {
    /// A translation fault.
    Translation,
    /// An access flag fault.
    AccessFlag,
    /// A permission fault.
    Permission,
    /// A TLB conflict abort.
    TlbConflict,
    /// Any other data abort, with its ESR.
    Other(u64),
}

impl Fault {
    fn from_esr(esr: u64) -> Self {
        // The data fault status code.
        match esr & 0x3f {
            0b000100..=0b000111 => Self::Translation,
            0b001000..=0b001011 => Self::AccessFlag,
            0b001100..=0b001111 => Self::Permission,
            0b110000 => Self::TlbConflict,
            _ => Self::Other(esr & !FAULTED),
        }
    }
}

impl Display for Fault {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Translation => write!(f, "translation fault"),
            Self::AccessFlag => write!(f, "access flag fault"),
            Self::Permission => write!(f, "permission fault"),
            Self::TlbConflict => write!(f, "TLB conflict abort"),
            Self::Other(esr) => write!(f, "data abort with ESR {esr:#x}"),
        }
    }
}

/// Loads the 64-bit value at `va`, and returns the fault instead if the load causes a data abort.
///
/// # Safety
///
/// `va` must be 8-byte aligned, and it must not map memory that the program uses in a way that
/// breaks Rust's aliasing rules.
pub unsafe fn read(va: usize) -> Result<u64, Fault> {
    // SAFETY: The probe function loads from `va`, which our caller promises to be fine, and the
    // exception handler makes it return the fault if the load causes a data abort.
    let result = unsafe { probe_read(va) };
    match result.esr {
        0 => Ok(result.value),
        esr => Err(Fault::from_esr(esr)),
    }
}

/// Stores `value` at `va`, and returns the fault instead if the store causes a data abort.
///
/// # Safety
///
/// `va` must be 8-byte aligned, and it must not map memory that the program uses in a way that
/// breaks Rust's aliasing rules.
pub unsafe fn write(va: usize, value: u64) -> Result<(), Fault> {
    // SAFETY: The probe function stores to `va`, which our caller promises to be fine, and the
    // exception handler makes it return the fault if the store causes a data abort.
    match unsafe { probe_write(va, value) } {
        0 => Ok(()),
        esr => Err(Fault::from_esr(esr)),
    }
}
