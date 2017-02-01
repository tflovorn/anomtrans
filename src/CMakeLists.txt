list(APPEND SOURCES
  MPIPrettyUnitTestResultPrinter.cpp
  special_functions.cpp
)

add_mpi_test(special_functions "${SOURCES}")
