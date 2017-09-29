list(APPEND SOURCES
  MPIPrettyUnitTestResultPrinter.cpp
  vec.cpp
  mat.cpp
  util.cpp
  special_functions.cpp
  square_tb_spectrum.cpp
  pauli.cpp
  Rashba_Hamiltonian.cpp
  rho0.cpp
  collision.cpp
  dm_graph.cpp
)

add_mpi_test(vec "${SOURCES}")
add_mpi_test(mat "${SOURCES}")
add_mpi_test(util "${SOURCES}")
add_mpi_test(special_functions "${SOURCES}")
add_mpi_test(grid_basis "${SOURCES}")
add_mpi_test(energy "${SOURCES}")
add_mpi_test(derivative "${SOURCES}")
add_mpi_test(square_tb_spectrum "${SOURCES}")
add_mpi_test(Rashba_Hamiltonian "${SOURCES}")
