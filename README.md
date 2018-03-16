# anomtrans [![Build Status](https://travis-ci.org/tflovorn/anomtrans.svg?branch=master)](https://travis-ci.org/tflovorn/anomtrans)

A library providing a numerical implementation of the density matrix formulation of quantum transport, including inter-band coherence and scattering from multiple Fermi surface sheets.

The formalism implemented here is presented in:

[Culcer, Sekine, and MacDonald, Phys. Rev. B 96, 035106 (2017)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.035106)

[Sekine, Culcer, and MacDonald, Phys. Rev. B 96, 235134 (2017)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.235134)

A paper discussing the numerical implementation provided by this repository is in preparation.

Regression and unit tests providing examples of the use of this software are given in the `test` directory. The following tests have been validated against analytic results (using the "result plots" parameter sets in the test files):

* The spin accumulation and spin Hall conductivity for the Rashba model calculated by the `Rashba_electric` test in `test/Rashba_Hamiltonian_test.cpp`.
* The chiral magnetic effect current for a single Weyl node calculated by the `wsm_continuum_cme_node` test in `test/wsm_continuum_Hamiltonian_test.cpp`.

Additionally the following tests are expected to be correct, but have not been fully validated:

* The anomalous Hall conductivity of the Rashba model with exchange field calculated by the `Rashba_magnetized_electric` test in `test/Rashba_magnetic_Hamiltonian_test.cpp` yields the expected cancellation of the intrinsic and extrinsic contributions, but the absolute value of these contributions has not been checked.
* The anomalous Hall conductivity of the Weyl semimetal calculated by the `wsm_continuum_ahe` test in `test/wsm_continuum_Hamiltonian_test.cpp` appears to converge toward the correct result as the k-point density and sampling volume are increased, but this convergence has not been fully checked.

This software is distributed under the MIT license to allow maximum freedom of use. However we ask that any publication making use of this software or derivative software cite the two papers listed above and acknowledge use of this repository, providing a link to it (after the numerical paper is available, we will ask that such publications cite that paper as well). This software is in active development, and while we are confident in the cases which have been validated as noted above, we make no general guarantee of correctness. We encourage users or potential users of this software to contact us via mailing list at [anomtrans@googlegroups.com](mailto:anomtrans@googlegroups.com), or to file issues in this repository with specific bug reports or feature suggestions.

# Local setup and usage

These instructions are based on a fresh Linux Mint 18.1 installation. They should work for any Debian-based distribution.

## Dependencies and Setup

Get g++, gfortran, CMake, OpenMPI, valgrind, boost, doxygen, matplotlib, scipy:

    sudo apt-get install g++ gfortran cmake libopenmpi-dev openmpi-bin valgrind libboost-all-dev doxygen graphviz python-matplotlib python-tk python3-matplotlib python3-tk python3-setuptools python3-scipy libubsan0 lib64ubsan0 curl

Note that the Boost package is Boost 1.58.

PETSc 3.8 is not available from the package manager. We'll need to build it. [(download page)](https://www.mcs.anl.gov/petsc/download/index.html) [(install instructions)](https://www.mcs.anl.gov/petsc/documentation/installation.html)

    cd ~
    git clone -b maint https://bitbucket.org/petsc/petsc petsc
    cd ~/petsc
    ./configure PETSC_ARCH=arch-linux2-cxx-complex-debug --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --with-clanguage=cxx --with-scalar-type=complex
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-complex-debug all
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-complex-debug test
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-complex-debug streams

`anomtrans` regression test data is stored in [Git LFS](https://github.com/git-lfs/git-lfs/releases). Install this:

    cd ~
    curl -L -o git-lfs-linux-amd64-2.3.4.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v2.3.4/git-lfs-linux-amd64-2.3.4.tar.gz
    tar -xvzf git-lfs-linux-amd64-2.3.4.tar.gz
    cd git-lfs-2.3.4
    PREFIX=$HOME ./install.sh
    export PATH=$HOME/bin:$PATH

To keep `git lfs` accessible, add its location to your default PATH. In the file `~/.bashrc`, add:

    export PATH=$HOME/bin:$PATH

With Git LFS installed, we can now clone the `anomtrans` repository and get the test data.

    cd ~
    git clone https://github.com/tflovorn/anomtrans.git

(If this repository was cloned before Git LFS was installed, the test data will not have been fetched. To remedy this, run `git lfs fetch` in the `anomtrans` directory.)

Get Google Test submodule:

    cd ~/anomtrans
    git submodule init
    git submodule update

To allow for generation of result plots, install pyanomtrans:

    cd ~/anomtrans
    python3 setup.py develop --user

## Usage

To build documentation:

    cd ~/anomtrans
    doxygen

To build and run tests (should be done from the anomtrans root directory):

    cd ~/anomtrans
    ./build_test_local
    cd Obj_test
    ctest -V

To generate plots from tests:

    cd ~/anomtrans/pyanomtrans
    python3 plot_Rashba.py "Rashba_Hamiltonian_test_out" "../Obj_test/src"
    python3 plot_wsm.py "wsm_continuum_cme_test_out" "../Obj_test/src"
    python3 plot_series.py (other test output file here) "../Obj_test/src"   
    python3 plot_2d_bz.py (other test output file here) "../Obj_test/src"

## Building in release mode

Build PETSc with:

    cd ~/petsc
    ./configure PETSC_ARCH=arch-linux2-cxx-complex-opt --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --with-clanguage=cxx --with-scalar-type=complex --with-debugging=0 COPTFLAGS='-O3 -march=native -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' FOPTFLAGS='-O3 -march=native -mtune=native'
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-complex-opt all
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-complex-opt test
    make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux2-cxx-complex-opt streams

Build and run tests:

    cd ~/anomtrans
    ./build_release_local
    cd Obj
    ctest -V

TODO - MKL BLAS/LAPACK support?

# Installation and usage on Lonestar5

## Dependencies

Assumes the intel/16.0.1 module is loaded on Lonestar5.

Add module dependencies to `~/.bashrc` just under "PLACE MODULE COMMANDS HERE and ONLY HERE.":

    module load cmake boost petsc/3.7-cxxcomplexdebug

Replace `petsc/3.7-cxxcomplexdebug` with `petsc/3.7-cxxcomplex` for release mode.

Also add to `~/.bashrc` just under "PLACE Environment Variables including PATH here.":

    export PATH=$HOME/bin:$PATH

On Lonestar5, it is not necessary to set `LIBRARY_PATH` or `LD_LIBRARY_PATH` for boost or petsc.
The correct `LD_LIBRARY_PATH` is set by module load.

anomtrans regression test data is stored in [Git LFS](https://github.com/git-lfs/git-lfs/releases). Install this:

    cd ~
    curl -L -o git-lfs-linux-amd64-2.3.4.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v2.3.4/git-lfs-linux-amd64-2.3.4.tar.gz
    tar -xvzf git-lfs-linux-amd64-2.3.4.tar.gz
    cd git-lfs-2.3.4
    PREFIX=$HOME ./install.sh
    export PATH=$HOME/bin:$PATH

With Git LFS installed, we can now clone the `anomtrans` repository and get the test data.

    cd ~
    git clone https://github.com/tflovorn/anomtrans.git

(If this repository was cloned before Git LFS was installed, the test data will not have been fetched. To remedy this, run `git lfs fetch` in the `anomtrans` directory.)

Get Google Test submodule:

    cd ~/anomtrans
    git submodule init
    git submodule update

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

# Development Notes

Exceptions thrown by `anomtrans` functions are intended to be a 'panic' type of error.
They should not be caught unless explicitly noted as being safe to catch.
The lack of RAII for PETSc objects makes exception safety cumbersome, and in general
it is not attempted here to provide this type of safety.

Whitespace conventions of the code are satisfied by the following `~/.vimrc`:

    filetype plugin indent on

    set tabstop=4
    set shiftwidth=4
    set expandtab

    syntax on

    au BufRead,BufNewFile Makefile* set noexpandtab

    au BufRead,BufNewFile *.cpp,*.h,CMakeLists.txt set tabstop=2
    au BufRead,BufNewFile *.cpp,*.h,CMakeLists.txt set shiftwidth=2

# Acknowledgements

This implementation has been developed at the University of Texas at Austin with support from the Department of Energy, Office of Basic Energy Sciences, under Contract No. DE-FG02-ER45958, and from the Welch foundation, under Grant No. TBF1473.
