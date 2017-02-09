list(APPEND SOURCES
  MPIPrettyUnitTestResultPrinter.cpp
  util.cpp
  special_functions.cpp
  square_tb_spectrum.cpp
  vec.cpp
  mat.cpp
  collision.cpp
)

add_mpi_test(special_functions "${SOURCES}")
add_mpi_test(grid_basis "${SOURCES}")
add_mpi_test(energy "${SOURCES}")
add_mpi_test(vec "${SOURCES}")
add_mpi_test(derivative "${SOURCES}")
