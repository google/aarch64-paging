# aarch64 page table manipulation

[![crates.io page](https://img.shields.io/crates/v/aarch64-paging.svg)](https://crates.io/crates/aarch64-paging)
[![docs.rs page](https://docs.rs/aarch64-paging/badge.svg)](https://docs.rs/aarch64-paging)

This crate provides a library to manipulate page tables conforming to the AArch64 Virtual Memory
System Architecture.

Currently it only supports:

- stage 1 page tables
- 4 KiB pages
- EL3, NS-EL2, NS-EL2&0 and NS-EL1&0 translation regimes

This is not an officially supported Google product.

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

If you want to contribute to the project, see details of
[how we accept contributions](CONTRIBUTING.md).
