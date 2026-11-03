// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Randomized tests of `aarch64-paging` with page tables that are actually live, in a bare-metal
//! virtual machine, i.e., QEMU's `virt` machine with TCG, or with KVM on an arm64 host.
//!
//! The tests run at EL1 with the EL1&0 translation regime, or at EL2 with the EL2&0 translation
//! regime, with VHE enabled, if the virtual machine starts at EL2 (`-M virt,virtualization=on`,
//! which needs nested virtualization support with KVM). See `fuzz.rs` for what they check.
//!
//! The kernel command line (`-append`) may contain these options:
//!
//! - `runs=<n>`: the number of sequences of updates per test variant;
//! - `ops=<n>`: the number of updates per sequence;
//! - `seed=<n>`: run only the sequence with this seed, which is printed before each sequence;
//! - `variant=lower|upper`: run only the variant for the lower or upper VA range;
//! - `verbose=1`: print the updates as they are made, rather than only when a test fails.

#![no_std]
#![no_main]

extern crate alloc;

#[macro_use]
mod console;
mod exceptions;
mod fdt;
mod fuzz;

// The shared part of the randomized tests refers to the crate under test via `crate::` paths.
use aarch64_paging::{MapError, Mapping, descriptor, paging};

use aarch64_paging::descriptor::El1Attributes;
use aarch64_paging::paging::{El1And0, El2And0};
use buddy_system_allocator::LockedHeap;
use core::arch::{asm, global_asm};
use core::panic::PanicInfo;

/// The size of each block of the initial identity map.
const BLOCK_SIZE: usize = 1 << 30;
/// The physical address range of the devices of the `virt` machine that are used.
const DEVICES: (usize, usize) = (0, 1 << 30);
/// The physical address range of RAM, assuming that the virtual machine has 4 GiB of it.
const RAM: (usize, usize) = (1 << 30, 5 << 30);
/// The size of the heap, which follows the image.
const HEAP_SIZE: usize = 64 << 20;

/// The attributes of the identity map of the devices.
const DEVICE_ATTRIBUTES: El1Attributes = El1Attributes::VALID
    .union(El1Attributes::ATTRIBUTE_INDEX_0)
    .union(El1Attributes::ACCESSED)
    .union(El1Attributes::PXN)
    .union(El1Attributes::UXN);
/// The attributes of the identity map of RAM.
const RAM_ATTRIBUTES: El1Attributes = El1Attributes::VALID
    .union(El1Attributes::ATTRIBUTE_INDEX_1)
    .union(El1Attributes::INNER_SHAREABLE)
    .union(El1Attributes::ACCESSED);

/// MAIR: Device-nGnRE, Normal write-back cacheable, and Normal non-cacheable memory.
const MAIR: u64 = 0x04 | 0xff << 8 | 0x44 << 16;
/// TCR, apart from the IPS field, with the ASID taken from TTBR0 (A1 = 0).
const TCR: u64 = 25 // T0SZ: 39-bit VA range
    | 0b01 << 8 // IRGN0: write-back, write-allocate
    | 0b01 << 10 // ORGN0: write-back, write-allocate
    | 0b11 << 12 // SH0: inner shareable
    | 25 << 16 // T1SZ: 39-bit VA range
    | 0b01 << 24 // IRGN1: write-back, write-allocate
    | 0b01 << 26 // ORGN1: write-back, write-allocate
    | 0b11 << 28 // SH1: inner shareable
    | 0b10 << 30; // TG1: 4 KiB granule
/// SCTLR, with the bits that are RES1 in Armv8.0 set.
const SCTLR: u64 = 1 << 0 // M: MMU enabled
    | 1 << 2 // C: data caches enabled
    | 1 << 3 // SA: stack alignment checks
    | 1 << 4 // SA0
    | 1 << 11 // EOS
    | 1 << 12 // I: instruction caches enabled
    | 1 << 20 // TSCXT
    | 1 << 22 // EIS
    | 1 << 23 // SPAN: PSTATE.PAN left alone on exception entry
    | 1 << 28 // nTLSMD
    | 1 << 29; // LSMAOE
/// HCR_EL2: E2H, TGE and RW.
const HCR_EL2: u64 = 1 << 34 | 1 << 27 | 1 << 31;
/// TCR.A1, which selects the ASID of TTBR1 rather than that of TTBR0.
const TCR_A1: u64 = 1 << 22;
const PSCI_SYSTEM_OFF: u64 = 0x8400_0008;

#[repr(C, align(4096))]
struct Table([usize; 512]);

/// The initial identity map, which maps the devices and RAM with level 1 block descriptors, in the
/// same way as the page tables of the tests for the lower VA range do.
static IDMAP: Table = {
    let mut table = [0; 512];
    let mut pa = DEVICES.0;
    while pa < DEVICES.1 {
        table[pa / BLOCK_SIZE] = pa | DEVICE_ATTRIBUTES.bits();
        pa += BLOCK_SIZE;
    }
    pa = RAM.0;
    while pa < RAM.1 {
        table[pa / BLOCK_SIZE] = pa | RAM_ATTRIBUTES.bits();
        pa += BLOCK_SIZE;
    }
    Table(table)
};

/// The initial page table for TTBR1, which maps nothing.
static EMPTY: Table = Table([0; 512]);

global_asm!(
    r#"
    .section .text.start, "ax"
    .global _start
_start:
    /*
     * If started at EL2, enable VHE with HCR_EL2.{{E2H, TGE}} = {{1, 1}}, so that the EL2&0
     * translation regime is used, and so that the EL1 system register names that are used from
     * here on refer to the corresponding EL2 registers.
     */
    mrs     x0, CurrentEL
    ubfx    x0, x0, #2, #2
    cmp     x0, #1
    b.eq    1f
    cmp     x0, #2
    b.ne    .Lunsupported
    mrs     x1, id_aa64mmfr1_el1
    ubfx    x1, x1, #8, #4              /* VH */
    cbz     x1, .Lunsupported
    ldr     x1, ={hcr_el2}
    msr     hcr_el2, x1
    isb
1:
    mrs     x1, id_aa64mmfr0_el1
    ubfx    x2, x1, #28, #4             /* TGran4 */
    cmp     x2, #0xf
    b.eq    .Lunsupported
    ubfx    x1, x1, #0, #4              /* PARange */
    mov     x2, #5                      /* 48 bits */
    cmp     x1, x2
    csel    x1, x1, x2, lo
    ldr     x2, ={tcr}
    bfi     x2, x1, #32, #3             /* IPS */
    msr     tcr_el1, x2
    ldr     x1, ={mair}
    msr     mair_el1, x1
    adrp    x1, {idmap}
    add     x1, x1, :lo12:{idmap}
    msr     ttbr0_el1, x1
    adrp    x1, {empty}
    add     x1, x1, :lo12:{empty}
    msr     ttbr1_el1, x1
    isb
    tlbi    vmalle1
    dsb     nsh
    isb
    ldr     x1, ={sctlr}
    msr     sctlr_el1, x1
    isb

    /* Enable FP/SIMD, which the compiler may use. */
    mov     x1, #(3 << 20)
    msr     cpacr_el1, x1
    isb

    adrp    x1, vector_table
    add     x1, x1, :lo12:vector_table
    msr     vbar_el1, x1
    isb

    adrp    x1, __stack_top
    add     x1, x1, :lo12:__stack_top
    mov     sp, x1

    adrp    x1, __bss_start
    add     x1, x1, :lo12:__bss_start
    adrp    x2, __bss_end
    add     x2, x2, :lo12:__bss_end
2:  cmp     x1, x2
    b.hs    3f
    stp     xzr, xzr, [x1], #16
    b       2b
3:
    bl      {main}

.Lunsupported:
    adr     x1, .Lunsupported_message
    mov     x2, #{uart}
4:  ldrb    w3, [x1], #1
    cbz     w3, 5f
    str     w3, [x2]
    b       4b
5:  ldr     x0, ={psci_system_off}
    mrs     x1, CurrentEL
    cmp     x1, #(1 << 2)
    b.ne    6f
    hvc     #0
6:  smc     #0
7:  wfi
    b       7b
.Lunsupported_message:
    .asciz  "Unsupported exception level or CPU features\r\ntest result: FAILED\r\n"
"#,
    hcr_el2 = const HCR_EL2,
    tcr = const TCR,
    mair = const MAIR,
    sctlr = const SCTLR,
    idmap = sym IDMAP,
    empty = sym EMPTY,
    main = sym main,
    uart = const console::UART_BASE,
    psci_system_off = const PSCI_SYSTEM_OFF,
);

#[global_allocator]
static HEAP: LockedHeap<32> = LockedHeap::empty();

/// Returns the current exception level.
fn current_el() -> u64 {
    let el: u64;
    // SAFETY: Reading CurrentEL has no side effects.
    unsafe { asm!("mrs {}, CurrentEL", out(reg) el, options(nomem, nostack)) };
    (el >> 2) & 3
}

/// Invalidates all TLB entries of the current translation regime on this CPU.
fn flush_tlb() {
    // SAFETY: TLB invalidation has no effect on the program, other than making it use the current
    // page tables.
    unsafe {
        asm!(
            "dsb ishst",
            "tlbi vmalle1",
            "dsb nsh",
            "isb",
            options(nostack)
        )
    };
}

/// Sets or clears TCR.A1, to use the ASID of TTBR1 or TTBR0, respectively.
fn set_tcr_a1(a1: bool) {
    let mut tcr: u64;
    // SAFETY: This only changes which ASID is used, and invalidates the TLB afterwards, as entries
    // may have been tagged with the other one.
    unsafe {
        asm!("mrs {}, tcr_el1", out(reg) tcr, options(nomem, nostack));
        tcr = if a1 { tcr | TCR_A1 } else { tcr & !TCR_A1 };
        asm!("msr tcr_el1, {}", "isb", in(reg) tcr, options(nostack));
    }
    flush_tlb();
}

/// The options given on the kernel command line.
pub struct Config {
    runs: usize,
    ops: usize,
    seed: Option<u64>,
    lower: bool,
    upper: bool,
    verbose: bool,
}

impl Config {
    fn parse(bootargs: &str) -> Self {
        let mut config = Self {
            runs: 8,
            ops: 300,
            seed: None,
            lower: true,
            upper: true,
            verbose: false,
        };
        for arg in bootargs.split_ascii_whitespace() {
            let (key, value) = arg.split_once('=').unwrap_or((arg, "1"));
            let number = || {
                match value.strip_prefix("0x") {
                    Some(hex) => u64::from_str_radix(hex, 16),
                    None => value.parse(),
                }
                .unwrap_or_else(|_| panic!("invalid value for {key}: {value}"))
            };
            match key {
                "runs" => config.runs = number() as usize,
                "ops" => config.ops = number() as usize,
                "seed" => config.seed = Some(number()),
                "variant" => {
                    config.lower = value == "lower";
                    config.upper = value == "upper";
                }
                "verbose" => config.verbose = number() != 0,
                _ => println!("ignoring unknown option {arg}"),
            }
        }
        config
    }
}

/// Turns the virtual machine off.
fn system_off() -> ! {
    let smc = match fdt::property("psci", "method") {
        Some(b"smc\0") => true,
        Some(b"hvc\0") => false,
        // QEMU uses the SMC conduit if EL2 is implemented, and the HVC one otherwise.
        _ => current_el() == 2,
    };
    // SAFETY: This doesn't return if it succeeds, and has no effect otherwise.
    unsafe {
        if smc {
            asm!("smc #0", inout("x0") PSCI_SYSTEM_OFF => _, options(nostack));
        } else {
            asm!("hvc #0", inout("x0") PSCI_SYSTEM_OFF => _, options(nostack));
        }
    }
    loop {
        // SAFETY: Waiting for an interrupt has no effect on the program.
        unsafe { asm!("wfi", options(nomem, nostack)) };
    }
}

extern "C" fn main() -> ! {
    console::Uart::init();
    unsafe extern "C" {
        static __image_end: u8;
    }
    // SAFETY: The heap is RAM that follows the image, which nothing else uses.
    unsafe {
        HEAP.lock().init(&raw const __image_end as usize, HEAP_SIZE);
    }
    let bootargs = fdt::property("chosen", "bootargs")
        .and_then(|value| core::str::from_utf8(value).ok())
        .map_or("", |value| value.trim_end_matches('\0'));
    let config = Config::parse(bootargs);
    // The memory node has two address cells and two size cells.
    let memory = fdt::property("memory", "reg").and_then(|reg| {
        let cell = |i: usize| {
            Some(u64::from_be_bytes(
                reg.get(i * 8..i * 8 + 8)?.try_into().ok()?,
            ))
        };
        Some((cell(0)? as usize, cell(0)?.checked_add(cell(1)?)? as usize))
    });
    match memory {
        Some((start, end)) if start <= RAM.0 && RAM.1 <= end => {}
        Some((start, end)) => panic!(
            "RAM must cover {:#x}..{:#x}, but it covers {start:#x}..{end:#x}",
            RAM.0, RAM.1
        ),
        None => panic!("No memory node found in the device tree"),
    }
    let el = current_el();
    let sequences = if el == 2 {
        println!("Running at EL2 with VHE, using the EL2&0 translation regime");
        fuzz::run::<El2And0>(&config)
    } else {
        println!("Running at EL1, using the EL1&0 translation regime");
        fuzz::run::<El1And0>(&config)
    };
    println!("test result: ok. {sequences} sequences passed");
    system_off()
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    console::dump_log();
    println!("{info}");
    println!("test result: FAILED");
    system_off()
}
