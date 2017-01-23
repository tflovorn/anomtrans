# Dependencies

Get Eigen:

    ./setup_eigen

Get Google Test:

    git submodule init
    git submodule update

Assumes the intel/16.0.1 module is loaded on Lonestar5.

Dependencies are available on Lonestar5 with

    module load boost/1.5.9 petsc/3.7-cxxcomplex

for release mode or

    module load boost/1.5.9 petsc/3.7-cxxcomplexdebug

for debug mode.

Set library paths in ~/.bashrc:

    export LIBRARY_PATH=$TACC_PETSC_LIB:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$TACC_PETSC_LIB:$LIBRARY_PATH

TODO - support local build.

# Building

To build documentation:

    doxygen

To build and run tests (should be done from the llblg root directory):

    ./run_tests

To build release version:

    ./build_release
