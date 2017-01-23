# Dependencies

Assumes the intel/16.0.1 module is loaded on Lonestar5.

Dependencies are available on Lonestar5 with:

    module load cmake boost trilinos

Note that the boost/1.59 module on Lonestar5 does not include boost::mpi.
`module help boost` references a boost-mpi module, but this does not exist.

On Lonestar5, it is not necessary to set LIBRARY_PATH or LD_LIBRARY_PATH for boost or trilinos.
The correct LD_LIBRARY_PATH is set by module load.

TODO - support local build.

# Building

To build documentation:

    doxygen

To build and run tests (should be done from the anomtrans root directory):

    ./build_test
    idev
    cd Obj_test/test
    ctest
    exit

For verbose test information, run ctest -V instead of ctest.

TODO - build release version.
