/*
 * Copyright 2026 The aarch64-paging Authors.
 * This project is dual-licensed under Apache 2.0 and MIT terms.
 * See LICENSE-APACHE and LICENSE-MIT for details.
 */

/*
 * QEMU loads the image at its link address, and puts the device tree at the start of RAM at
 * 0x4000_0000 if it fits below the image.
 */
ENTRY(_start)

SECTIONS
{
	. = 0x40200000;
	.text : {
		KEEP(*(.text.start))
		*(.text .text.*)
	}
	.rodata : ALIGN(4096) {
		*(.rodata .rodata.*)
	}
	.data : ALIGN(4096) {
		*(.data .data.*)
	}
	.bss (NOLOAD) : ALIGN(4096) {
		__bss_start = .;
		*(.bss .bss.*)
		*(COMMON)
		. = ALIGN(16);
		__bss_end = .;
	}
	.stack (NOLOAD) : ALIGN(4096) {
		. += 0x100000;
		__stack_top = .;
	}
	/* The heap follows the image. */
	. = ALIGN(4096);
	__image_end = .;

	/DISCARD/ : {
		*(.comment)
		*(.eh_frame .eh_frame_hdr)
		*(.note .note.*)
	}
}
