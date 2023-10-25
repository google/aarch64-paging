# Changelog

## 0.x.x

### Bug fixes

- Reject the `PAGE_OR_TABLE` flag when passed to `map_range`, which would result
  in corrupt table mappings to be created.

### Breaking changes

- Updated `modify_range` to split block entries before traversing them, and
  pass only the descriptors and subregions that are completely covered by the
  given region to the updater callback function.
- Updated `modify_range` to only pass block or page descriptors to the callback
  function and prevent them from being converted into table descriptors
  inadvertently.
- Added rigid break-before-make (BBM) checks to `map_range` and `modify_range`.
- Added `constraints` argument to new `map_range()` alternative
  `map_range_with_constraints()`.
- Marked `activate` and `deactivate` methods as unsafe.

### New features

- Added `walk_range` method that iterates over all block or page descriptors
  that intersect with a given region, without permitting the callback to make
  changes to the descriptors

## 0.4.1

### Bug fixes

- `RootTable`, `Mapping`, `IdMap` and `LinearMap` are now correctly marked as `Send`, as it doesn't
  matter where they are used from.

## 0.4.0

### Breaking changes

- Updated `bitflags` to 2.0.2, which changes the API of `Attributes` a bit.
- Updated `map_range` method to support mapping leaf page table entries without the `VALID` flag.
  `Attributes::VALID` is no longer implicitly set when mapping leaf page table entries.

### New features

- Added `modify_range` method to `IdMap`, `LinearMap` and `Mapping` to update details of a mapped
  range. This can be used e.g. to change flags for some range which is already mapped. As part of
  this, the `Descriptor` struct was added to the public API.
- Added `DBM` and software flags to `Attributes`.

## 0.3.0

### Breaking changes

- Made `Translation` trait responsible for allocating page tables. This should help make it possible
  to use more complex mapping schemes, and to construct page tables in a different context to where
  they are used.
- Renamed `AddressRangeError` to `MapError`, which is now an enum with three variants and implements
  `Display`.
- `From<*const T>` and `From<*mut T>` are no longer implemented for `VirtualAddress`.
- Added support for using TTBR1 as well as TTBR0; this changes various constructors to take an extra
  parameter.

### New features

- Made `alloc` dependency optional via a feature flag.
- Added support for linear mappings with new `LinearMap`.
- Implemented subtraction of usize from address types.

### Bugfixes

- Fixed memory leak introduced in 0.2.0: dropping a page table will now actually free its memory.

## 0.2.1

### New features

- Implemented `Debug` and `Display` for `MemoryRegion`.
- Implemented `From<Range<VirtualAddress>>` for `MemoryRegion`.
- Implemented arithmetic operations for `PhysicalAddress` and `VirtualAddress`.

## 0.2.0

### Breaking changes

- Added bounds check to `IdMap::map_range`; it will now return an error if you attempt to map a
  virtual address outside the range of the page table given its configured root level.

### New features

- Implemented `Debug` for `PhysicalAddress` and `VirtualAddress`.
- Validate that chosen root level is supported.

### Bugfixes

- Fixed bug in `Display` and `Drop` implementation for `RootTable` that would result in a crash for
  any pagetable with non-zero mappings.
- Fixed `Display` implementation for `PhysicalAddress` and `VirtualAddress` to use correct number of
  digits.

## 0.1.0

Initial release.
