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
set(EIGEN_INCLUDE_DIR "dependencies/Eigen")

# System-provided dependencies.
set(PETSC_INCLUDE_DIR "$ENV{PETSC_DIR}/include")
set(BOOST_INCLUDE_DIR $ENV{BOOST_INC})
set(PETSC_ARCH_INCLUDE_DIR "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include")

# Build gtest.
add_subdirectory(${GOOGLETEST_DIR})

# Set include directories for dependencies.
include_directories(${GOOGLETEST_INCLUDE_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${PETSC_INCLUDE_DIR})
include_directories(${PETSC_ARCH_INCLUDE_DIR})
include_directories(SYSTEM ${JSON_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})

# Libraries to link.
add_library(petsc SHARED IMPORTED)
set_target_properties(petsc PROPERTIES IMPORTED_LOCATION "$ENV{PETSC_LIB}/libpetsc.so")
set(ANOMTRANS_TEST_LIBS gtest gtest_main petsc)

enable_testing()

# Unit testing with MPI programs, following the approach at:
# https://scicomp.stackexchange.com/a/8739
function(add_mpi_test name other_sources)
  add_executable(${name} ${name}_test.cpp ${other_sources})
  # TODO - express this in compiler-independent way?
  # This is the set for icpc 16 (could also add -Wremarks).
  # For gcc 5+, would use:
  # -Wall -pedantic -std=c++11"
  # (or -std=c++14, but would want to keep compatibility with icpc 16)
  set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} $ENV{ANOMTRANS_TEST_COMPILE_FLAGS}"
  )
  target_link_libraries(${name} ${ANOMTRANS_TEST_LIBS})
  set(test_parameters $ENV{ANOMTRANS_TEST_MPI_NUMPROCS_PREFIX} $ENV{ANOMTRANS_TEST_MPI_NUMPROCS} $ENV{ANOMTRANS_TEST_AFFINITY} "./${name}")
  add_test(NAME ${name} COMMAND $ENV{ANOMTRANS_TEST_MPI_CMD} ${test_parameters})
  set_property(TEST ${name} PROPERTY
    ENVIRONMENT "ANOMTRANS_TEST_DATA_DIR=$ENV{ANOMTRANS_TEST_DATA_DIR}")
endfunction(add_mpi_test)

add_subdirectory(src)
