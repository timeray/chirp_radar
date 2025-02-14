cmake_minimum_required(VERSION 3.18)

project(Tutorial 
    VERSION 0.0.1
    LANGUAGES C CXX CUDA
    DESCRIPTION "Libraries and programs to process radar data for studies in solar-terrestrial physics"
    )
add_executable(ChirpProc src/chirp_proc.cpp src/gpu_ops/core.cu)

#add_executable(TestIseIo tests/test_ise_io.cpp src/stp_radar/iseio.cpp)
add_executable(TestIseIo tests/test_ise_io.cpp)

add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_20)

target_link_libraries(TestIseIo PUBLIC tutorial_compiler_flags gtest_main)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)

target_include_directories(ChirpProc PUBLIC src)
