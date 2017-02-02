list(APPEND SOURCES
  MPIPrettyUnitTestResultPrinter.cpp
  vec.cpp
  special_functions.cpp
  square_tb_spectrum.cpp
)

add_mpi_test(special_functions "${SOURCES}")
add_mpi_test(grid_basis "${SOURCES}")
add_mpi_test(energy "${SOURCES}")
