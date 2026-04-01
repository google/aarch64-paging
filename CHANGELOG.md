# Changelog

## 0.12.1

### Bugfixes

- Fixed value of `El23Attributes::XN`.

### Improvements

- Added `NT` and `CONTIGUOUS_OR_PROTECTED` bits to `El1Attributes` and `El23Attributes`.
- Added `El23Attributes::USER_RES1`.

## 0.12.0

### Breaking changes

- `Attributes` has been split into `El1Attributes`, `El23Attributes` and `Stage2Attributes`. Use the
  appropriate type for the translation regime you are using. This has been added as a generic
  parameter to `Descriptor`, `PageTable`, `TargetAllocator`, `IdTranslation`, `LinearTranslation`
  and `UpdatableDescriptor`.
- `TranslationRegime` is now a trait rather than an enum, with implementations for each regime. This
  has been added as a generic parameter to `RootTable`, `Mapping`, `IdMap` and `LinearMap`.
- `MapError::InvalidFlags` has a `usize` rather than an `Attributes` value.
- `Mapping`, `IdMap` and `LinearMap` now have separate constructors for regimes with (`with_asid` or
  `with_asid_and_va_range`) and without (`new`) an ASID.
- `IdTranslation` is no longer a unit struct; it can be constructed with `IdTranslation::new`.

### New features

- Added support for stage-2 page tables.

### Bugfixes

- Don't use blocks at level 0. This doesn't work unless LPA2 is enabled, which we don't yet support.

## 0.11.0

### Breaking changes

- Manipulation of live descriptors now performs the required TLB maintenance after each update so
  that the attribute changes take effect immediately. To achieve this, the `modify_range` methods on
  `IdMap`, `LinearMap` and `Mapping` now pass the new `UpdatableDescriptor` type rather than
  `Descriptor` and level to the updater function.
- `RootTable::modify_range` is no longer public.

### New features

- Added `compact_subtables` method to `IdMap`, `LinearMap`, `Mapping` and `RootTable`. This will
  replace any subtables which are entirely empty with an invalid descriptor, and free the subtables.

### Improvements

- `Debug` implementation for `RootTable` now includes `translation_regime` and `va_range` fields.
- `Debug` implementation for `RootTable` now groups contiguous mapped ranges together.
- Unmapping a region covering an entire subtable will replace the table descriptor with an invalid
  descriptor and free the subtable.
- Unmapping a region will zero the PA, and won't add the `TABLE_OR_PAGE` bit to the attributes
  given.

## 0.10.0

### New features

- Added `Attributes::GP` bit for BTI guarded pages.

### Breaking changes

- `zerocopy` feature has been removed. `PageTable::write_to` is provided instead.
- `IdMap::activate` now returns the previous TTBR value rather than storing it, and
  `IdMap::deactivate` takes the TTBR value to restore as a parameter. `IdMap::mark_active` no longer
  takes a previous TTBR value parameter. The same applies to the equivalent methods on `LinearMap`.
- Renamed `Mapping::activate_raw` to `activate`, and added previous TTBR value parameter to
  `Mapping::deactivate`.
- A page table may be activated multiple times (e.g. on multiple cores) and will keep track of how
  many times it has been activated. It will only be considered inactive once it has been deactivated
  the same number of times.
- `MapError::PteUpdateFault` now contains a `usize` rather than a `Descriptor`.
- `Descriptor` no longer implements `Copy`, `Clone`, `Default`, `PartialEq` or `Eq`, as it now
  contains an `AtomicUsize` rather than just a `usize`. Various methods on `Descriptor` now take
  `&self` rather than `self`.

## 0.9.1

### New features

- Added `Mapping::activate_raw` to support multicore usage and other scenarios.
- Added `Mapping::asid` to get ASID.

## 0.9.0

### Breaking changes

- `Descriptor::flags` now returns `Attributes` rather than `Option<Attributes>`. If any unknown bits
  are set they will be included.

### Bug fixes

- `Descriptor::is_table_or_page` will return the correct value even if unknown bits are set.
  Previously it would return false in this case.

### New features

- Added `PXN_TABLE`, `XN_TABLE`, `AP_TABLE_NO_EL0`, `AP_TABLE_NO_WRITE` and `NS_TABLE` bits to
  `Attributes`.

## 0.8.1

### New features

- `MapError` now implements `core::error::Error`.
- Made `output_address` method on `Descriptor` public.

## 0.8.0

### Breaking changes

- Updated to `zerocopy` 0.8.2.

### New features

- Added `mair` module with types to build MAIR values.
- Implemented `Default` for `Attributes`, `Descriptor`, `PageTable`, `PhysicalAddress` and
  `VirtualAddress`. Added associated `EMPTY` constant for `Descriptor` and `PageTable`.

## 0.7.1

### New features

- Added `Mapping::translation` method.
- Added `Mapping::size`, `IdMap::size` and `LinearMap::size` methods.
- Derived zerocopy traits for `VirtualAddress`, `PhysicalAddress`, `PageTable` and `Descriptor`.
  This is guarded behind the `zerocopy` feature so the dependency can be avoided if not desired.
- Added `TargetAllocator` for pregenerating a static pagetable for a target device.

## 0.7.0

### Breaking changes

- `Translation::allocate_table` and `Translation::deallocate_table` now takes `&mut self` rather
  than `&self.

### Other changes

- The `Translation` type parameter to `Mapping` no longer needs to be `Clone`.
- `IdMap`, `LinearMap`, `Mapping` and `RootTable` are now `Sync`.

## 0.6.0

### Breaking changes

- Added support for EL2 and EL3 page tables. This requires a new parameter to `IdMap::new`,
  `LinearMap::new`, `Mapping::new` and `RootTable::new`.
- `Attributes::EXECUTE_NEVER` renamed to `Attributes::UXN`.
- `Attributes::DEVICE_NGNRE` and `NORMAL` have been removed in favour of `ATTRIBUTE_INDEX_*`,
  `OUTER_SHAREABLE` and `INNER_SHAREABLE`, to avoid making assumptions about how the MAIR registers
  are programmed.

### New features

- Added `root_address`, `mark_active` and `mark_inactive` methods to `IdMap`, `LinearMap` and
  `Mapping`. These may be used to activate and deactivate the page table manually rather than
  calling `activate` and `deactivate`.
- Added `NS` and `PXN` bits to `Attributes`.

### Bug fixes

- When an invalid descriptor is split into a table, the table descriptors aren't set unless to
  non-zero values unless the original descriptor was.

### Other changes

- `Attributes::ACCESSED` is no longer automatically set on all new mappings. To maintain existing
  behaviour you should explicitly set `Attributes::ACCESSED` whenever calling `map_range` for a
  valid mapping.

## 0.5.0

### Bug fixes

- Reject the `PAGE_OR_TABLE` flag when passed to `map_range`, which would result in corrupt table
  mappings to be created.

### Breaking changes

- Updated `modify_range` to split block entries before traversing them, and pass only the
  descriptors and subregions that are completely covered by the given region to the updater callback
  function.
- Updated `modify_range` to only pass block or page descriptors to the callback function and prevent
  them from being converted into table descriptors inadvertently.
- Added rigid break-before-make (BBM) checks to `map_range` and `modify_range`.
- Marked `activate` and `deactivate` methods as unsafe.

### New features

- Added new `map_range()` alternative `map_range_with_constraints()` with extra `contraints`
  argument.
- Added `walk_range` method that iterates over all block or page descriptorsthat intersect with a
  given region, without permitting the callback to make changes to the descriptors

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
