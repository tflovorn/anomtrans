set(SOURCES
  special_functions.cpp
)

add_mpi_test(special_functions ${SOURCES})
add_mpi_test(grid_basis ${SOURCES})
