#!/bin/sh
# Copyright 2026 The aarch64-paging Authors.
# This project is dual-licensed under Apache 2.0 and MIT terms.
# See LICENSE-APACHE and LICENSE-MIT for details.
#
# Runs the tests in QEMU, as the runner of `cargo run`, which passes the path of the image and any
# arguments after `--`, which become the kernel command line, e.g. `cargo run --release -- runs=8`.
#
# Environment variables:
#   QEMU:           the QEMU binary, `qemu-system-aarch64` by default
#   VMTEST_EL:      `2` to run at EL2, with VHE, rather than at EL1
#   VMTEST_KVM:     `1` to use KVM rather than TCG, on an arm64 host (with nested virtualization
#                   support for EL2)
#   VMTEST_TIMEOUT: the timeout in seconds, 3600 by default

set -eu

image=$1
shift

machine=virt,gic-version=max
if [ "${VMTEST_EL:-1}" = 2 ]; then
	machine=$machine,virtualization=on
fi
if [ "${VMTEST_KVM:-0}" = 1 ]; then
	accel="-accel kvm -cpu host"
else
	accel="-accel tcg -cpu max"
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT

# The image expects 4 GiB of RAM.
timeout "${VMTEST_TIMEOUT:-3600}" "${QEMU:-qemu-system-aarch64}" \
	-M "$machine" $accel -smp 1 -m 4G -nodefaults -display none -serial stdio \
	-kernel "$image" -append "$*" </dev/null | tee "$log"

grep -q '^test result: ok' "$log"
