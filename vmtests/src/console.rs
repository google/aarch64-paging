// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! Output via the PL011 UART of QEMU's `virt` machine, and a log that is only printed if a test
//! fails.

use core::fmt::{self, Write};
use core::ptr::{read_volatile, write_volatile};
use spin::mutex::SpinMutex;

pub const UART_BASE: usize = 0x0900_0000;
const UART_DR: usize = 0x00;
const UART_FR: usize = 0x18;
const UART_FR_TXFF: u32 = 1 << 5;
const UART_CR: usize = 0x30;
const UART_CR_UARTEN: u32 = 1 << 0;
const UART_CR_TXE: u32 = 1 << 8;

/// The PL011 UART, whose baud rate etc. don't matter in QEMU.
pub struct Uart;

impl Uart {
    /// Enables the UART for transmission.
    pub fn init() {
        // SAFETY: As for `putc`.
        unsafe {
            write_volatile(
                (UART_BASE + UART_CR) as *mut u32,
                UART_CR_UARTEN | UART_CR_TXE,
            );
        }
    }

    fn putc(c: u8) {
        // SAFETY: The UART registers are identity mapped as device memory, and nothing else uses
        // them.
        unsafe {
            while read_volatile((UART_BASE + UART_FR) as *const u32) & UART_FR_TXFF != 0 {}
            write_volatile((UART_BASE + UART_DR) as *mut u32, c.into());
        }
    }
}

impl Write for Uart {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.bytes() {
            if c == b'\n' {
                Self::putc(b'\r');
            }
            Self::putc(c);
        }
        Ok(())
    }
}

macro_rules! println {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!($crate::console::Uart, $($arg)*);
    }};
}

/// The size of the log, which keeps the most recent output.
const LOG_SIZE: usize = 64 << 10;

/// A ring buffer with the most recent lines of the log.
struct Log {
    buffer: [u8; LOG_SIZE],
    /// The number of bytes written so far.
    written: usize,
    verbose: bool,
}

impl Write for Log {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.bytes() {
            self.buffer[self.written % LOG_SIZE] = c;
            self.written += 1;
        }
        Ok(())
    }
}

static LOG: SpinMutex<Log> = SpinMutex::new(Log {
    buffer: [0; LOG_SIZE],
    written: 0,
    verbose: false,
});

/// Makes `log` print lines right away rather than when a test fails.
pub fn set_verbose(verbose: bool) {
    LOG.lock().verbose = verbose;
}

/// Adds a line to the log.
pub fn log(line: fmt::Arguments) {
    let mut log = LOG.lock();
    if log.verbose {
        println!("{line}");
    } else {
        let _ = writeln!(log, "{line}");
    }
}

/// Prints the most recent lines of the log, unless they have been printed already.
pub fn dump_log() {
    // The panic may have occurred while the log was locked.
    let Some(log) = LOG.try_lock() else {
        return;
    };
    if log.verbose || log.written == 0 {
        return;
    }
    let start = log.written.saturating_sub(LOG_SIZE);
    let mut bytes = (start..log.written).map(|i| log.buffer[i % LOG_SIZE]);
    if start > 0 {
        println!("[... earlier lines of the log omitted ...]");
        // Skip the rest of the oldest line, which is incomplete.
        bytes.by_ref().find(|&c| c == b'\n');
    }
    for c in bytes {
        if c == b'\n' {
            Uart::putc(b'\r');
        }
        Uart::putc(c);
    }
}
