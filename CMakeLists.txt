cmake_minimum_required(VERSION 3.29)
project(cleanfm)

set(CMAKE_CXX_STANDARD 20)

# Manually set up FFTW3
add_library(FFTW3::FFTW3 SHARED IMPORTED)
set_target_properties(FFTW3::FFTW3 PROPERTIES
        IMPORTED_LOCATION "/opt/homebrew/Cellar/fftw/3.3.10_2/lib/libfftw3.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "/opt/homebrew/Cellar/fftw/3.3.10_2/include"
)

# Library target
add_library(cleanfm_lib
        ${CMAKE_CURRENT_SOURCE_DIR}/fft/waveprocessor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fft/waveprocessor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/constants.h
        ${CMAKE_CURRENT_SOURCE_DIR}/deconvolve/dirichletkernel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/deconvolve/dirichletkernel.h
        deconvolve/cleandft.cpp
        deconvolve/cleandft.h
)

# Set include directories for the library
target_include_directories(cleanfm_lib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

# Main executable
add_executable(cleanfm main.cpp)

# Link libraries to the executable
target_link_libraries(cleanfm
        PRIVATE
        cleanfm_lib
        FFTW3::FFTW3
)

# Ensure dr_wav implementation is included
target_compile_definitions(cleanfm_lib
        PRIVATE
        DR_WAV_IMPLEMENTATION
)