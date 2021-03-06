list(APPEND SOURCES
  util/MPIPrettyUnitTestResultPrinter.cpp
  util/vec.cpp
  util/mat.cpp
  util/util.cpp
  util/special_functions.cpp
  models/square_tb_spectrum.cpp
  models/Rashba_Hamiltonian.cpp
  models/wannier90_Hamiltonian.cpp
  models/tmd_nn_Hamiltonian.cpp
  models/wsm_continuum_Hamiltonian.cpp
  models/wsm_continuum_node_Hamiltonian.cpp
  models/wsm_continuum_mu5_Hamiltonian.cpp
  observables/energy.cpp
  observables/spin.cpp
  observables/rho0.cpp
  dyn_dm_graph.cpp
  fermi_surface.cpp
)

add_library(anomtrans STATIC ${SOURCES})
set_target_properties(anomtrans PROPERTIES
  COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${ANOMTRANS_OpenMP_CXX_FLAGS} $ENV{ANOMTRANS_COMPILE_FLAGS}"
  LINK_FLAGS "${ANOMTRANS_OpenMP_CXX_FLAGS}"
)
target_link_libraries(anomtrans ${ANOMTRANS_LIBS})
