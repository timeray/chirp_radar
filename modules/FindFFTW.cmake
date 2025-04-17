set(FFTW_ROOT_MSG "FFTW_ROOT cmake variable can be used to set arbitrary path to FFTW.")

find_path(FFTW_INCLUDE_DIR NAMES fftw3.h)
if (NOT FFTW_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find FFTW headers. ${FFTW_ROOT_MSG}")
endif()

message("FFTW3 header directory: ${FFTW_INCLUDE_DIR}")

find_library(FFTW_LIBRARY NAMES fftw3)
if (NOT FFTW_LIBRARY)
    message(FATAL_ERROR "Could not find FFTW3 libraries. ${FFTW_ROOT_MSG}")
endif()

find_library(FFTWF_LIBRARY NAMES fftw3f PATHS)
if(NOT FFTWF_LIBRARY)
    message(FATAL_ERROR "Could not find FFTW3F libraries. ${FFTW_ROOT_MSG}")
endif()

set(FFTW_LIBRARIES ${FFTW_LIBRARY} ${FFTWF_LIBRARY})
message("FFTW3 libraries: ${FFTW_LIBRARIES}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    FFTW DEFAULT_MSG
    FFTW_LIBRARIES FFTW_INCLUDE_DIR
)
mark_as_advanced(FFTW_INCLUDE_DIR FFTW_LIBRARIES)
