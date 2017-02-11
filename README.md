# Installation and usage on Lonestar5

## Dependencies

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

On Lonestar5, it is not necessary to set `LIBRARY_PATH` or `LD_LIBRARY_PATH` for boost or petsc.
The correct `LD_LIBRARY_PATH` is set by module load.

## Usage

To build documentation:

    doxygen

To build and run tests (should be done from the anomtrans root directory):

    ./build_test_ls5
    idev
    cd Obj_test
    ctest -V
    exit

Or to submit a job to test:

    ./build_test_ls5
    sbatch run_tests_ls5
    exit

TODO - build release version.

# Local setup and usage from a fresh Mint 18.1 MATE install

## Basics

Get Vim:

    sudo apt-get install vim

Set up ~/.vimrc -- `vim ~/.vimrc`, then add:

    filetype plugin indent on

    set tabstop=4
    set shiftwidth=4
    set expandtab

    syntax on

    au BufRead,BufNewFile Makefile* set noexpandtab

    au BufRead,BufNewFile *.cpp,*.h,CMakeLists.txt set tabstop=2
    au BufRead,BufNewFile *.cpp,*.h,CMakeLists.txt set shiftwidth=2

## Dependencies

Get g++, gfortran, CMake, OpenMPI, valgrind, matplotlib, boost, doxygen:

    sudo apt-get install g++ gfortran cmake libopenmpi-dev openmpi-bin valgrind python-matplotlib python-tk libboost-all-dev doxygen graphviz

Note that the Boost package is Boost 1.58.

PETSc 3.7 is not available from the package manager. We'll need to build it. [(download page)](https://www.mcs.anl.gov/petsc/download/index.html) [(install instructions)](https://www.mcs.anl.gov/petsc/documentation/installation.html)

    cd ~
    git clone -b maint https://bitbucket.org/petsc/petsc petsc
    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --with-clanguage=cxx
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-debug all

Test PETSc:

    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-debug test
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-debug streams

## Usage

To build documentation:

    doxygen

To build and run tests (should be done from the anomtrans root directory):

    ./build_test_local
    cd Obj_test
    ctest -V

TODO - build release version.
