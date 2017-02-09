# Dependencies

Get Google Test:

    git submodule init
    git submodule update

Assumes the intel/16.0.1 module is loaded on Lonestar5.

Dependencies are available on Lonestar5 with

    module load cmake boost petsc/3.7-cxx

for release mode or

    module load cmake boost petsc/3.7-cxxdebug

for debug mode.

Note that the boost/1.59 module on Lonestar5 does not include boost::mpi.
`module help boost` references a boost-mpi module, but this does not exist.

On Lonestar5, it is not necessary to set LIBRARY_PATH or LD_LIBRARY_PATH for boost or petsc.
The correct LD_LIBRARY_PATH is set by module load.

TODO - support local build.

# Building

To build documentation:

    doxygen

To build and run tests (should be done from the anomtrans root directory):

    ./build_test
    idev
    cd Obj_test
    ctest -V
    exit

Or to submit a job to test:

    ./build_test
    sbatch run_tests_ls5
    exit

TODO - build release version.
