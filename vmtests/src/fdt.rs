// Copyright 2026 The aarch64-paging Authors.
// This project is dual-licensed under Apache 2.0 and MIT terms.
// See LICENSE-APACHE and LICENSE-MIT for details.

//! A minimal parser for the flattened device tree that QEMU passes, to find the kernel command
//! line and the PSCI conduit.

/// The address at which QEMU puts the device tree for an ELF image that is loaded above it.
const FDT_ADDRESS: usize = 0x4000_0000;
const FDT_MAGIC: u32 = 0xd00d_feed;
const FDT_BEGIN_NODE: u32 = 1;
const FDT_END_NODE: u32 = 2;
const FDT_PROP: u32 = 3;
const FDT_NOP: u32 = 4;

/// Returns the device tree blob, if there is one.
fn blob() -> Option<&'static [u8]> {
    let read = |offset: usize| {
        // SAFETY: The start of RAM is identity mapped and not used by the program, and QEMU puts
        // the device tree there if it fits.
        u32::from_be(unsafe { ((FDT_ADDRESS + offset) as *const u32).read_volatile() })
    };
    if read(0) != FDT_MAGIC {
        return None;
    }
    let size = read(4) as usize;
    // SAFETY: As above, and the header says how large the device tree is.
    Some(unsafe { core::slice::from_raw_parts(FDT_ADDRESS as *const u8, size) })
}

fn be32(blob: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_be_bytes(
        blob.get(offset..offset + 4)?.try_into().unwrap(),
    ))
}

/// Returns the NUL-terminated string at `offset`, without the NUL.
fn string(blob: &[u8], offset: usize) -> Option<&[u8]> {
    let bytes = blob.get(offset..)?;
    Some(&bytes[..bytes.iter().position(|&c| c == 0)?])
}

/// Returns the value of the property called `name` of the child node of the root node called
/// `node`.
pub fn property(node: &str, name: &str) -> Option<&'static [u8]> {
    let blob = blob()?;
    let structs = be32(blob, 8)? as usize;
    let strings = be32(blob, 12)? as usize;
    let mut offset = structs;
    let mut depth = 0;
    let mut in_node = false;
    loop {
        let token = be32(blob, offset)?;
        offset += 4;
        match token {
            FDT_BEGIN_NODE => {
                let node_name = string(blob, offset)?;
                offset = (offset + node_name.len() + 1).next_multiple_of(4);
                depth += 1;
                // Ignore the unit address, if any.
                let base_name = node_name.split(|&c| c == b'@').next()?;
                in_node = depth == 2 && base_name == node.as_bytes();
            }
            FDT_END_NODE => {
                depth -= 1;
                in_node = false;
            }
            FDT_PROP => {
                let len = be32(blob, offset)? as usize;
                let name_offset = be32(blob, offset + 4)? as usize;
                let value = blob.get(offset + 8..offset + 8 + len)?;
                offset = (offset + 8 + len).next_multiple_of(4);
                if in_node && string(blob, strings + name_offset)? == name.as_bytes() {
                    return Some(value);
                }
            }
            FDT_NOP => {}
            _ => return None,
        }
    }
}
