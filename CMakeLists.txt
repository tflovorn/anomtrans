project(anomtrans)
cmake_minimum_required(VERSION 2.6.4) # minimum for googletest

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})

# Dependencies inside the project directory.
set(JSON_INCLUDE_DIR "dependencies/json/src")
set(GOOGLETEST_DIR "dependencies/googletest")
set(GOOGLETEST_INCLUDE_DIR "dependencies/googletest/googletest/include")

# Include paths on Lonestar5.
# TODO should provide mechanism for local implementation.
set(PETSC_INCLUDE_DIR "$ENV{TACC_PETSC_DIR}/include")
set(BOOST_INCLUDE_DIR $ENV{TACC_BOOST_INC})
set(PETSC_ARCH_INCLUDE_DIR "$ENV{TACC_PETSC_DIR}/$ENV{PETSC_ARCH}/include")

# Build gtest.
add_subdirectory(${GOOGLETEST_DIR})

# Set include directories for dependencies.
include_directories(${GOOGLETEST_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${PETSC_INCLUDE_DIR})
include_directories(${PETSC_ARCH_INCLUDE_DIR})
include_directories(${JSON_INCLUDE_DIR})

# Libraries to link.
add_library(petsc SHARED IMPORTED)
set_target_properties(petsc PROPERTIES IMPORTED_LOCATION "$ENV{TACC_PETSC_LIB}/libpetsc.so")
set(ANOMTRANS_TEST_LIBS gtest gtest_main petsc)

# Lonestar5 MPI parameters:
# TODO should provide mechanism for local implementation.
set(MPIEXEC "ibrun")
set(MPIEXEC_NUMPROC_FLAG "-np")

enable_testing()

# Unit testing with MPI programs, following the approach at:
# https://scicomp.stackexchange.com/a/8739
# Modified for use with Trilinos testing framework.
function(add_mpi_test name other_sources)
  add_executable(${name} ${name}_test.cpp ${other_sources})
  # TODO - express this in compiler-independent way?
  # This is the set for icpc (could also add -Wremarks).
  # For gcc 5+, would use:
  # -Wall -pedantic -std=c++14"
  set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11"
  )
  target_link_libraries(${name} ${ANOMTRANS_TEST_LIBS})
  set(test_parameters ${MPIEXEC_NUMPROC_FLAG} $ENV{ANOMTRANS_TEST_PROCS} "./${name}")
  add_test(NAME ${name} COMMAND ${MPIEXEC} ${test_parameters})
endfunction(add_mpi_test)

add_subdirectory(src)
