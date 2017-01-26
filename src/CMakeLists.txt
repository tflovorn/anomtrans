set(SOURCES
  special_functions.cpp
  special_functions.h
  dist_vec.h
  grid_basis.h
)

add_mpi_test(special_functions ${SOURCES})
add_mpi_test(grid_basis ${SOURCES})
