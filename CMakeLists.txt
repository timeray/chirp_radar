cmake_minimum_required(VERSION 3.18)

project(Tutorial 
    VERSION 0.0.1
    LANGUAGES CXX CUDA
    DESCRIPTION "Libraries and programs to process radar data for studies in solar-terrestrial physics"
    )

find_package(CUDAToolkit REQUIRED)

add_executable(ChirpProc src/chirp_proc.cu)
add_executable(TestIseIo tests/test_ise_io.cpp)
add_executable(TestFFT src/gpu_ops/tests/test_basic_chirp_fft.cu)
add_executable(BenchFFT src/gpu_ops/tests/bench_basic_chirp_fft.cu)
add_executable(BenchFFTEvents src/gpu_ops/tests/bench_basic_chirp_fft_events.cu)

add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_20)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
add_subdirectory(external/benchmark EXCLUDE_FROM_ALL)
#add_subdirectory(external/armadillo-code EXCLUDE_FROM_ALL)
add_subdirectory(external/fftw EXCLUDE_FROM_ALL)

# Required for legacy callbacks in cufft (linking against static library)
set_target_properties(TestFFT PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                         CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(BenchFFT PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                          CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(BenchFFTEvents PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                                CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ChirpProc PUBLIC tutorial_compiler_flags CUDA::cufft)
target_link_libraries(TestIseIo PUBLIC tutorial_compiler_flags gtest_main)
# Linking against static cufft library for legacy cufft callbacks
target_link_libraries(TestFFT PUBLIC tutorial_compiler_flags gtest_main CUDA::cufft_static fftw3 fftw3f)
target_link_libraries(BenchFFT PUBLIC tutorial_compiler_flags benchmark_main CUDA::cufft_static fftw3 fftw3f)
target_link_libraries(BenchFFTEvents PUBLIC tutorial_compiler_flags benchmark_main CUDA::cufft_static fftw3 fftw3f)

target_include_directories(ChirpProc PUBLIC src)
target_include_directories(TestFFT PUBLIC src)
target_include_directories(BenchFFT PUBLIC src)
target_include_directories(BenchFFTEvents PUBLIC src)
