FROM ubuntu:xenial as anomtrans_base

MAINTAINER Tim Lovorn, tflovorn@gmail.com

# Install package dependencies for PETSc.
RUN apt-get update && \
    apt-get install -y \
    g++ \
    gfortran \
    cmake \
    libopenmpi-dev \
    openmpi-bin \
    git \
    python

# Build latest version of PETSc, debug version.
WORKDIR /

RUN git clone -b maint https://bitbucket.org/petsc/petsc petsc

WORKDIR /petsc

RUN ./configure PETSC_ARCH=arch-linux2-cxx-complex-debug --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-fblaslapack --with-clanguage=cxx --with-scalar-type=complex && \
    make PETSC_DIR=/petsc PETSC_ARCH=arch-linux2-cxx-complex-debug all

ENV PETSC_DIR=/petsc

# Install package dependencies for anomtrans.
# TODO - can we avoid installing all of Boost here, picking only select packages?
#     Documentation is not obvious about how all the pieces are divided up.
RUN apt-get update && \
    apt-get install -y \
    libboost-all-dev \
    libubsan0

# Build anomtrans.
# TODO - is it possible to re-copy and re-build anomtrans when we do `docker run`,
#     in order to rebuild only what is necessary?
FROM anomtrans_base
COPY . /anomtrans

WORKDIR /anomtrans

ENV ANOMTRANS_DIR=/anomtrans
ENV ANOMTRANS_MPI_ALLOW_ROOT=1

RUN ./build_test_local

# Run tests.
CMD cd /anomtrans/Obj_test && ctest -V
