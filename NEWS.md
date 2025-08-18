# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added options to optimise the state map update for `AffineFEStateMap` and `NonlinearFEStateMap`
- Added transient tests

### Fixed
- Bug fix in `StateParamMaps` to correctly use analytic gradient

## [0.4.0] - 2025-7-15

### Added
- Added `Evolver` and `Reinitialiser` as part of full `LevelSetEvolution` refactor.
- Added `HeatReinitialiser` based on Feng and Crane (2024) [doi: 10.1145/3658220]. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/81), similarly below.
- Added `IdentityReinitialiser`, does nothing.
- Added `get_element_diameters` methods of QUAD and HEX.

### Changed
- Refactored caching in StateMaps to remove constructor dependence on primal variable.
- Refactored `LevelSetEvolution` to split evolution and reinitialisation method.
- Deprecated `γ_reinit` from optimiser options.
- Deprecated `StateParamIntegrandWithMeasure` with an error.
- Warning when passing `U_reg` to state maps has been replaced with an error to fully deprecate methods.
- Disabled out-of-date methods in Benchmarks.
- Overhauled Breaking Changes section of Docs

## [0.3.0] - 2025-7-4

### Added
- Backwards AD via Zygote is now supported in serial and parallel. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/80).

### Changed
- StateMaps now always differentiate into a consistent space. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/80).
- Removed `U_reg` space from StateMaps. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/80).
- Refactored allocation of vectors in distributed. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/80).

### Fixed
- Resolved Issue[#46](https://github.com/zjwegert/GridapTopOpt.jl/issues/46)

## [0.2.2] - 2025-6-19

### Fixed
- Minor fixes and doc updates. As of PR[#77](https://github.com/zjwegert/GridapTopOpt.jl/pull/77).

## [0.2.1] - 2025-6-19

### Fixed
- Minor fixes and doc updates.

## [0.2.0] - 2025-6-18

### Added
- Added compatibility with GridapEmbedded for unfitted level-set topology optimisation. As of PR[#75](https://github.com/zjwegert/GridapTopOpt.jl/pull/75) and similarly below.
- Added isolated volume detection via polytopal cutting.
- Added embedded collections for updating embedded triangulations.
- Added unfitted evolution and reinitialisation methods.
- Added StaggeredFEStateMap for computing adjoints for problems involving [staggered FE problems](https://github.com/gridap/GridapSolvers.jl/blob/main/src/BlockSolvers/StaggeredFEOperators.jl).
- Added tests with finite differences for all StateMaps

### Changed
- Split ChainRules.jl into StateMaps/...
- Removed `IntegrandWithMeasure` and requirement to pass measures as arguments