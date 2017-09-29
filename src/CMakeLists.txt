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

add_library(anomtrans STATIC "${SOURCES}")
set_target_properties(anomtrans PROPERTIES
  COMPILE_FLAGS "${CMAKE_CXX_FLAGS} $ENV{ANOMTRANS_COMPILE_FLAGS}"
)
target_link_libraries(anomtrans ${ANOMTRANS_LIBS})

add_mpi_test(vec)
add_mpi_test(mat)
add_mpi_test(util)
add_mpi_test(special_functions)
add_mpi_test(grid_basis)
add_mpi_test(energy)
add_mpi_test(derivative)
add_mpi_test(square_tb_spectrum)
add_mpi_test(Rashba_Hamiltonian)
