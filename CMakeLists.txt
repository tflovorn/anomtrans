project(anomtrans)
cmake_minimum_required(VERSION 2.6.4) # minimum for googletest

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})

# Dependencies inside the project directory.
set(EIGEN3_INCLUDE_DIR "dependencies/eigen")
set(JSON_INCLUDE_DIR "dependencies/json/src")
set(GOOGLETEST_DIR "dependencies/googletest")

# Include paths on Lonestar5.
# TODO should provide mechanism for local implementation.
set(PETSC_INCLUDE_DIR "${TACC_PETSC_DIR}/include")
set(BOOST_INCLUDE_DIR ${TACC_BOOST_INC})

add_subdirectory(${GOOGLETEST_DIR})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${PETSC_INCLUDE_DIR})
include_directories(${JSON_INCLUDE_DIR})

# Lonestar5 MPI parameters:
# TODO should provide mechanism for local implementation.
set(MPIEXEC "ibrun")
set(MPIEXEC_NUMPROC_FLAG "-np")

# TODO libraries to link.
# MPI linking should be handled by using mpicxx.
set(ANOMTRANS_TEST_LIBS "petsc")

# Unit testing with MPI programs, following the approach at:
# https://scicomp.stackexchange.com/a/8739
function(add_mpi_test name)
  add_executable(${name} ${name}_test.cpp)
  add_dependencies(${name} googletest)
  # TODO - express this in compiler-independent way?
  # This is the set for icpc.
  # For gcc 5+, would use:
  # -Wall -pedantic -std=c++14"
  set_target_properties(all_tests PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wremarks -std=c++11"
  )
  target_link_libraries(${name} ${ANOMTRANS_TEST_LIBS})
  set(test_parameters ${MPIEXEC_NUMPROC_FLAG} ${ANOMTRANS_TEST_PROCS} "./${name}")
  add_test(NAME ${name} COMMAND ${MPIEXEC} ${test_parameters})
endfunction(add_mpi_test)

add_subdirectory(src)
