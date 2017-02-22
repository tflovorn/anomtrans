# Installation and usage on Lonestar5

## Dependencies

Get Google Test:

    git submodule init
    git submodule update

Assumes the intel/16.0.1 module is loaded on Lonestar5.

Add module dependencies to `~/.bashrc` just under "PLACE MODULE COMMANDS HERE and ONLY HERE.":

    module load cmake boost petsc/3.7-cxxdebug

Replace `petsc/3.7-cxxdebug` with `petsc/3.7-cxx` for release mode.

Note that the boost/1.59 module on Lonestar5 does not include boost::mpi.
`module help boost` references a boost-mpi module, but this does not exist.

Also add to `~/.bashrc` just under "PLACE Environment Variables including PATH here.":

    export PATH=$HOME/bin:$PATH

On Lonestar5, it is not necessary to set `LIBRARY_PATH` or `LD_LIBRARY_PATH` for boost or petsc.
The correct `LD_LIBRARY_PATH` is set by module load.

anomtrans regression test data is stored in [Git LFS](https://github.com/git-lfs/git-lfs/releases). Install this:

    cd ~
    curl -L -o git-lfs-linux-amd64-1.5.6.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v1.5.6/git-lfs-linux-amd64-1.5.6.tar.gz
    tar -xvzf git-lfs-linux-amd64-1.5.6.tar.gz
    cd git-lfs-1.5.6
    PREFIX=$HOME ./install.sh

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

To build and test in release mode:

    ./build_release_ls5
    idev
    cd Obj
    ctest -V
    exit

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

Get g++, gfortran, CMake, OpenMPI, valgrind, boost, doxygen, matplotlib, scipy:

    sudo apt-get install g++ gfortran cmake libopenmpi-dev openmpi-bin valgrind libboost-all-dev doxygen graphviz python-matplotlib python-tk python3-matplotlib python3-tk python3-setuptools python3-scipy

Note that the Boost package is Boost 1.58.

PETSc 3.7 is not available from the package manager. We'll need to build it. [(download page)](https://www.mcs.anl.gov/petsc/download/index.html) [(install instructions)](https://www.mcs.anl.gov/petsc/documentation/installation.html)

    cd ~
    git clone -b maint https://bitbucket.org/petsc/petsc petsc
    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --with-clanguage=cxx
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-debug all

Test PETSc:

    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-debug test
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-debug streams

anomtrans regression test data is stored in [Git LFS](https://github.com/git-lfs/git-lfs/releases). Install this:

    cd ~
    curl -L -o git-lfs-linux-amd64-1.5.6.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v1.5.6/git-lfs-linux-amd64-1.5.6.tar.gz
    tar -xvzf git-lfs-linux-amd64-1.5.6.tar.gz
    cd git-lfs-1.5.6
    PREFIX=$HOME ./install.sh

## Setup

Install pyanomtrans:

    cd ~/anomtrans
    python3 setup.py develop --user

## Usage

To build documentation:

    doxygen

To build and run tests (should be done from the anomtrans root directory):

    ./build_test_local
    cd Obj_test
    ctest -V

To generate plots from tests:

    cd pyanomtrans
    python3 plot_2d_bz.py "derivative_test_out" "../Obj_test/src"

## Building in release mode

Build PETSc with:

    cd ~/petsc
    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --with-clanguage=cxx --with-debugging=0 COPTFLAGS='-O3 -march=native -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' FOPTFLAGS='-O3 -march=native -mtune=native'
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-opt all

Test PETSc:

    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-opt test
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-opt streams

Build and run tests:

    cd ~/anomtrans
    ./build_release_local
    cd Obj
    ctest -V

TODO - MKL BLAS/LAPACK support?
