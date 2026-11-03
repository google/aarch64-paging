# Tests in a virtual machine

This crate runs the randomized tests of `aarch64-paging` (see `src/fuzz.rs`) in a bare-metal
virtual machine, with page tables that are actually live. In addition to checking the page tables
against a model of the expected mappings, it accesses the pages that each update changes, before
and after the update, and checks that the accesses fault, or load the expected data, as the model
says they should. This finds missing or incorrect TLB maintenance if the TLB happens to hold stale
entries, and incorrect descriptor attributes, but it is unlikely to find break-before-make
violations, which the TLB emulation of the tests on the host does find.

The tests run in QEMU's `virt` machine, either with TCG on any host, or with KVM on an arm64 host,
at EL1 with the EL1&0 translation regime, or at EL2 with the EL2&0 translation regime with VHE.

```sh
cd vmtests
# At EL1, with TCG.
cargo run --release
# At EL2 with VHE, with TCG.
VMTEST_EL=2 cargo run --release
# At EL1 or EL2, with KVM on an arm64 host. EL2 requires nested virtualization support, i.e.,
# `kvm-arm.mode=nested`, and a recent QEMU.
VMTEST_KVM=1 cargo run --release
VMTEST_KVM=1 VMTEST_EL=2 cargo run --release
# More, or longer, sequences of updates, or only the sequence with a given seed.
cargo run --release -- runs=100 ops=1000
cargo run --release -- seed=0x8cd06d6e783d43be variant=lower verbose=1
```

This needs the `aarch64-unknown-none` Rust target and `qemu-system-aarch64`, or the binary given in
`QEMU`. See `run.sh` and `src/main.rs` for more options.

QEMU's TLB doesn't behave like that of real hardware: it doesn't cache table descriptors, TLB
maintenance takes effect immediately rather than when it is completed by a DSB, and all TLB entries
are flushed on ASID changes and by `TLBI ASIDE1`. So running the tests with KVM is more likely to
find issues.
