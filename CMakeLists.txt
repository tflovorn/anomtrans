project(anomtrans)
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})

# Dependencies inside the project directory.
set(JSON_INCLUDE_DIR "dependencies/json/src")

# Include paths on Lonestar5.
# TODO should provide mechanism for local implementation.
#set(TRILINOS_PATH $ENV{TACC_TRILINOS_DIR})
set(BOOST_INCLUDE_DIR $ENV{TACC_BOOST_INC})
### These Trilinos paths should be set by FIND_PACKAGE(Trilinos ...), but this command doesn't work.
set(Trilinos_INCLUDE_DIRS "$ENV{TACC_TRILINOS_DIR}/include")
set(Trilinos_LIBRARY_DIRS $ENV{TACC_TRILINOS_LIB})

# Trilinos setup.
# Example Trilinos CMake file:
# https://github.com/trilinos/Trilinos_tutorial/wiki/CMakeFindPackageTrilinosExample
### The FIND_PACKAGE command here fails since the following file does not exist:
### /opt/apps/intel16/cray_mpich_7_3/trilinos/12.4.2/lib/cmake/STK/../STKDoc_tests/STKDoc_testsConfig.cmake
### There is an issue reported related to this:
### https://github.com/trilinos/Trilinos/issues/218

#FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})

#IF(Trilinos_FOUND)
#   MESSAGE("\nFound Trilinos!  Here are the details: ")
#   MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
#   MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
#   MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
#   MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
#   MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
#   MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
#   MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
#   MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
#   MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
#   MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
#   MESSAGE("   Trilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
#   MESSAGE("   Trilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
#   MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
#   MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
#   MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS =
#     ${Trilinos_Fortran_COMPILER_FLAGS}")
#   MESSAGE("   Trilinos_LINKER = ${Trilinos_LINKER}")
#   MESSAGE("   Trilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
#   MESSAGE("   Trilinos_AR = ${Trilinos_AR}")
#   MESSAGE("End of Trilinos details\n")
#ELSE()
#  MESSAGE(FATAL_ERROR "Could not find Trilinos!")
#ENDIF()
#
#IF(NOT CMAKE_CXX_COMPILER)
#  MESSAGE("Setting CXX compiler.")
#  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
#ENDIF()
#
#IF(NOT CMAKE_C_COMPILER)
#  MESSAGE("Setting C compiler.")
#  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
#ENDIF()
#

### TODO - remove this when FIND_PACKAGE(Trilinos ...) works.
### The following is copied from /opt/apps/intel16/cray_mpich_7_3/trilinos/12.4.2/lib/cmake/Trilinos/TrilinosConfig.cmake.
### Removed pytrilinos from Trilinos_LIBRARIES (link fails with this included due to missing Python C interface functions).
### Don't expect to use this.
### `module load python` seems to add the right Python interface library path to LD_LIBRARY_PATH, but not LIBRARY_PATH.
## The project libraries.
SET(Trilinos_LIBRARIES "sundancePdeopt;sundanceStdFwk;sundanceStdMesh;sundanceCore;sundanceInterop;sundanceUtils;sundancePlaya;stokhos_muelu;stokhos_ifpack2;stokhos_amesos2;stokhos_tpetra;stokhos_sacado;stokhos;moochothyra;moocho;rythmos;muelu-adapters;muelu-interface;muelu;locathyra;locaepetra;localapack;loca;noxepetra;noxlapack;nox;phalanx;stk_mesh_fixtures;stk_search;stk_unit_test_utils;stk_io_util;stk_io;stk_mesh_base;stk_topology;stk_util_use_cases;stk_util_registry;stk_util_parallel;stk_util_diag;stk_util_env;stk_util_util;stkclassic_adapt;stkclassic_percept;stkclassic_search_util;stkclassic_search;stkclassic_rebalance_utils;stkclassic_rebalance;stkclassic_linsys;stkclassic_io_util;stkclassic_io;stkclassic_expreval;stkclassic_algsup;stkclassic_mesh_fixtures;stkclassic_mesh_fem;stkclassic_mesh_base;stkclassic_util_use_cases;stkclassic_util_unit_test_support;stkclassic_util_parallel;stkclassic_util_diag;stkclassic_util_env;stkclassic_util_util;stk_mesh_fixtures;stk_search;stk_unit_test_utils;stk_io_util;stk_io;stk_mesh_base;stk_topology;stk_util_use_cases;stk_util_registry;stk_util_parallel;stk_util_diag;stk_util_env;stk_util_util;stkclassic_adapt;stkclassic_percept;stkclassic_search_util;stkclassic_search;stkclassic_rebalance_utils;stkclassic_rebalance;stkclassic_linsys;stkclassic_io_util;stkclassic_io;stkclassic_expreval;stkclassic_algsup;stkclassic_mesh_fixtures;stkclassic_mesh_fem;stkclassic_mesh_base;stkclassic_util_use_cases;stkclassic_util_unit_test_support;stkclassic_util_parallel;stkclassic_util_diag;stkclassic_util_env;stkclassic_util_util;intrepid;intrepid;teko;fei_trilinos;fei_base;stratimikos;stratimikosbelos;stratimikosaztecoo;stratimikosamesos;stratimikosml;stratimikosifpack;ifpack2-adapters;ifpack2;zoltan2;anasazitpetra;ModeLaplace;anasaziepetra;anasazi;belostpetra;belosepetra;belos;ml;komplex;ifpack;mapvarlib;fastqlib;blotlib;plt;svdi_cgi;svdi_cdr;suplib;supes;aprepro_lib;chaco;Ionit;Iotr;Iohb;Iogn;Iopg;Ioexo_fac;Iopx;Iofx;Ioex;Ioss;nemesis;exodus_for;exodus;mapvarlib;fastqlib;blotlib;plt;svdi_cgi;svdi_cdr;suplib;supes;aprepro_lib;chaco;Ionit;Iotr;Iohb;Iogn;Iopg;Ioexo_fac;Iopx;Iofx;Ioex;Ioss;nemesis;exodus_for;exodus;pamgen_extras;pamgen;amesos2;amesos;galeri-xpetra;galeri;aztecoo;dpliris;isorropia;xpetra-sup;xpetra;thyratpetra;thyraepetraext;thyraepetra;thyracore;thyratpetra;thyraepetraext;thyraepetra;thyracore;epetraext;tpetraext;tpetrainout;tpetra;kokkostsqr;tpetrakernels;tpetraclassiclinalg;tpetraclassicnodeapi;tpetraclassic;tpetraext;tpetrainout;tpetra;kokkostsqr;tpetrakernels;tpetraclassiclinalg;tpetraclassicnodeapi;tpetraclassic;triutils;shards;zoltan;epetra;sacado;rtop;teuchoskokkoscomm;teuchoskokkoscompat;teuchosremainder;teuchosnumerics;teuchoscomm;teuchosparameterlist;teuchoscore;teuchoskokkoscomm;teuchoskokkoscompat;teuchosremainder;teuchosnumerics;teuchoscomm;teuchosparameterlist;teuchoscore;kokkosalgorithms;kokkoscontainers;kokkoscore;kokkosalgorithms;kokkoscontainers;kokkoscore;tpi;gtest")

## The project tpl include paths
SET(Trilinos_TPL_INCLUDE_DIRS "/usr/include;/opt/apps/intel16/boost/1.59/include;/opt/apps/intel16/cray_mpich_7_2/parallel-netcdf/4.3.3.1/x86_64/include;/opt/apps/intel16/cray_mpich_7_2/phdf5/1.8.16/x86_64/include;/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/mkl/include")

## The project tpl library paths
SET(Trilinos_TPL_LIBRARY_DIRS "")

## The project tpl libraries
SET(Trilinos_TPL_LIBRARIES "/usr/lib64/libX11.so;/opt/apps/intel16/boost/1.59/lib/libboost_program_options.so;/opt/apps/intel16/boost/1.59/lib/libboost_system.so;/opt/apps/intel16/cray_mpich_7_2/parallel-netcdf/4.3.3.1/x86_64/lib/libnetcdf.so;/opt/apps/intel16/cray_mpich_7_2/phdf5/1.8.16/x86_64/lib/libhdf5.so;/usr/lib64/libz.so;/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_core.so;/usr/lib64/libpthread.so;/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_intel_lp64.so;/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_sequential.so;/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/libmkl_core.so;/usr/lib64/libpthread.so")

### End copy from /opt/apps/intel16/cray_mpich_7_3/trilinos/12.4.2/lib/cmake/Trilinos/TrilinosConfig.cmake.

# Set include directories and library directories for dependencies.
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${JSON_INCLUDE_DIR})

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

# Lonestar5 MPI parameters:
# TODO should provide mechanism for local implementation.
set(MPIEXEC "ibrun")
set(MPIEXEC_NUMPROC_FLAG "-np")

# TODO libraries to link.
# MPI linking should be handled by using mpicxx.
set(ANOMTRANS_TEST_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

enable_testing()

# Unit testing with MPI programs, following the approach at:
# https://scicomp.stackexchange.com/a/8739
# Modified for use with Trilinos testing framework.
function(add_mpi_test name)
  add_executable(${name} UnitTestMain.cpp ${name}_test.cpp)
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
