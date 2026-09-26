@echo off
setlocal

set "ROOT=%~dp0"

if "%EIGEN3_INCLUDE_DIR%"=="" (
  echo Set EIGEN3_INCLUDE_DIR to the directory containing Eigen\Dense.
  exit /b 1
)
if not exist "%EIGEN3_INCLUDE_DIR%\Eigen\Dense" (
  echo Eigen\Dense was not found under "%EIGEN3_INCLUDE_DIR%".
  exit /b 1
)

if "%SUITESPARSE_INCLUDE_DIR%"=="" (
  echo Set SUITESPARSE_INCLUDE_DIR to the directory containing cholmod.h.
  exit /b 1
)
if not exist "%SUITESPARSE_INCLUDE_DIR%\cholmod.h" (
  echo cholmod.h was not found under "%SUITESPARSE_INCLUDE_DIR%".
  exit /b 1
)

if "%CUDA_ARCH%"=="" set "CUDA_ARCH=sm_89"
if "%NVCC_EXTRA_FLAGS%"=="" set "NVCC_EXTRA_FLAGS=--use_fast_math -maxrregcount=152"

where nvcc >nul 2>nul
if errorlevel 1 (
  echo nvcc is not on PATH.
  exit /b 1
)

set "VCVARS="
if exist "D:\VS\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=D:\VS\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS (
  call "%VCVARS%" >nul
)

where cl >nul 2>nul
if errorlevel 1 (
  echo cl is not on PATH and Visual Studio 2022 Build Tools were not found.
  exit /b 1
)

set "MSVC_LIB="
for /d %%D in ("D:\VS\2022\BuildTools\VC\Tools\MSVC\*") do (
  if exist "%%~fD\lib\x64\libcmt.lib" set "MSVC_LIB=%%~fD\lib\x64"
)
for /d %%D in ("%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*") do (
  if exist "%%~fD\lib\x64\libcmt.lib" set "MSVC_LIB=%%~fD\lib\x64"
)
if not defined MSVC_LIB (
  echo Could not find the MSVC x64 library directory containing libcmt.lib.
  exit /b 1
)

set "SDK_UCRT="
set "SDK_UM="
for /d %%D in ("%ProgramFiles(x86)%\Windows Kits\10\Lib\*") do (
  if exist "%%~fD\ucrt\x64\ucrt.lib" set "SDK_UCRT=%%~fD\ucrt\x64"
  if exist "%%~fD\um\x64\uuid.lib" set "SDK_UM=%%~fD\um\x64"
)
if not defined SDK_UCRT (
  echo Could not find the Windows SDK x64 UCRT library directory.
  exit /b 1
)
if not defined SDK_UM (
  echo Could not find the Windows SDK x64 UM library directory.
  exit /b 1
)

for %%I in ("%SDK_UCRT%") do set "SDK_UCRT_SHORT=%%~sI"
for %%I in ("%SDK_UM%") do set "SDK_UM_SHORT=%%~sI"

if not exist "%ROOT%bin" mkdir "%ROOT%bin"

nvcc -std=c++17 -O3 -arch=%CUDA_ARCH% %NVCC_EXTRA_FLAGS% ^
  -I"%EIGEN3_INCLUDE_DIR%" ^
  -I"%SUITESPARSE_INCLUDE_DIR%" ^
  -Xcompiler /openmp ^
  "%ROOT%se2_cuda_hybrid_hgbp.cu" ^
  -Xlinker /LIBPATH:%MSVC_LIB% ^
  -Xlinker /LIBPATH:%SDK_UCRT_SHORT% ^
  -Xlinker /LIBPATH:%SDK_UM_SHORT% ^
  cusolver.lib ^
  cublas.lib ^
  -o "%ROOT%bin\se2_cuda_hybrid_hgbp.exe"

if errorlevel 1 exit /b 1

set "VC_REDIST="
for /d %%D in ("D:\VS\2022\BuildTools\VC\Redist\MSVC\*") do (
  if exist "%%~fD\x64\Microsoft.VC143.OpenMP\vcomp140.dll" set "VC_REDIST=%%~fD"
)
for /d %%D in ("%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC\*") do (
  if exist "%%~fD\x64\Microsoft.VC143.OpenMP\vcomp140.dll" set "VC_REDIST=%%~fD"
)
if defined VC_REDIST (
  copy /Y "%VC_REDIST%\x64\Microsoft.VC143.OpenMP\vcomp140.dll" "%ROOT%bin\" >nul
  if exist "%VC_REDIST%\x64\Microsoft.VC143.CRT\vcruntime140.dll" copy /Y "%VC_REDIST%\x64\Microsoft.VC143.CRT\vcruntime140.dll" "%ROOT%bin\" >nul
)

exit /b 0
