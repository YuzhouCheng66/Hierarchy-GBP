# Dependencies and runtime

The validated build uses MSVC 14.44, AVX2, Eigen 3.4.0, SuiteSparse Config
7.8.3 / CHOLMOD 5.3.0, and OpenBLAS 0.3.29. The CMake project preserves the
validated floating-point options: SE2's residual kernel uses `/fp:fast`;
SE3 does not. Other platforms/toolchains have not been release-validated.

```text
<dependency-root>/
  external/eigen/
  vcpkg_installed/x64-windows/{include,lib,bin}/
  external_baselines/rootba/
    src/ and external/
    build-win/src/rootba/Release/
    conda_env/Library/{include,lib,bin}/
```

RootBA reference revision: `d3900037fc4bc98328e5d8d26f1c4eed922f88cc`.
BA reuses its BAL data handling and linearization support, not its linear
solver as a substitute for H-GBP.

PGO requires a shared OpenBLAS **LP64** runtime exporting the LAPACKE
`dsbevx_work`, `dsbtrd_work`, and `dstemr_work` functions. A minimal BLAS-only
DLL is not interchangeable. The validated runtime reports
`OpenBLAS 0.3.29 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64` and its
`openblas.dll` SHA-256 is
`3fdccaf0054074f5979613a883b094db90a57395dc7ea31d5fd818c226ba2a1f`.

The official [OpenBLAS 0.3.29 Windows x64 archive](https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29_x64.zip)
provides an LP64 build. Do not use the `_x64_64` (ILP64) variant. Check the
runtime exports and numerical tests when substituting a different build.

Pass a directory containing OpenBLAS and its companion DLLs using
`-PGORuntimeDir` or `HGBP_PGO_RUNTIME`. `runtime/openblas_banded.def` builds
the Windows import library. The script stages vcpkg's SuiteSparse/LAPACK
dependencies first, then this OpenBLAS runtime into `build/pgo`.
BA receives its TBB/fmt/glog/gflags runtime dependencies plus the vcpkg
numerical DLLs in `build/ba`; it does not shadow the system MSVC/OpenMP DLLs
with conda copies. The build does not modify the dependency installations.

For explicit locations, the build script accepts `-EigenDir`,
`-VcpkgInstalledDir` and `-RootBADir`. PGO-only users may omit RootBA with
`-PGOOnly` (CMake: `-DHGBP_BUILD_BA=OFF`).

Dataset checksums and preprocessing are specified in `configs/datasets.json`.
Absolute paths in local validation records are not required by the release.
