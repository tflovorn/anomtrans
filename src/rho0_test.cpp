#include <cstddef>
#include <cmath>
#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <json.hpp>
#include "mpi.h"
#include "dist_vec.h"
#include "grid_basis.h"
#include "square_tb_spectrum.h"
#include "energy.h"
#include "rho0.h"
#include "derivative.h"

using json = nlohmann::json;

namespace {

TEUCHOS_UNIT_TEST( rho0, square_TB_fermi_surface ) {
  std::cout << "starting up" << std::endl;
  std::cout.flush();

  Teuchos::oblackholestream blackHole;
  anomtrans::MPIComm comm = anomtrans::get_comm();

  const int my_rank = comm->getRank();

  using GO = anomtrans::GO;
  using LO = anomtrans::LO;
  const std::size_t k_dim = 2;

  std::array<unsigned int, k_dim> Nk = {128, 128};
  unsigned int Nbands = 1;
  anomtrans::kmBasis<k_dim> kmb(Nk, Nbands, comm);
  auto kmb_map = kmb.get_map();

  std::cout << "made map" << std::endl;
  std::cout.flush();

  double t = 1.0;
  double tp = -0.3;
  anomtrans::square_tb_Hamiltonian H(t, tp, Nk);

  double beta = 10.0*t;

  anomtrans::DistVec<double> Ekm = anomtrans::get_energies(kmb, H);

  double E_F = 0.0; // TODO scan E_F from min(E) to max(E)

  anomtrans::DistVec<double> rho0 = anomtrans::make_rho0(kmb, Ekm, beta, E_F);

  std::cout << "got rho0" << std::endl;
  std::cout.flush();

  const unsigned int deriv_order = 2;
  auto d_dk = anomtrans::make_d_dk_recip(kmb, deriv_order);

  std::cout << "made d_dk" << std::endl;
  std::cout.flush();

  std::vector<anomtrans::DistVec<double>> d_rho0_dk;
  for (std::size_t d = 0; d < k_dim; d++) {
    d_rho0_dk.push_back(anomtrans::DistVec<double>(kmb_map));
    // d_rho0_dk(d) = [d_dk]_{d} * rho0
    d_dk.at(d)->apply(rho0, d_rho0_dk.at(d));
  }

  std::cout << "made d_rho0_dk" << std::endl;
  std::cout.flush();

  anomtrans::DistVec<double> norm_d_rho0_dk(kmb_map);
  norm_d_rho0_dk.sync<Kokkos::HostSpace>();
  auto norm_d_rho0_dk_2d = norm_d_rho0_dk.getLocalView<Kokkos::HostSpace>();
  auto norm_d_rho0_dk_1d = Kokkos::subview(norm_d_rho0_dk_2d, Kokkos::ALL(), 0);

  std::vector<decltype(norm_d_rho0_dk_2d)> d_rho0_dk_2d;
  std::vector<decltype(norm_d_rho0_dk_1d)> d_rho0_dk_1d;
  for (std::size_t d = 0; d < k_dim; d++) {
    d_rho0_dk.at(d).sync<Kokkos::HostSpace>();
    d_rho0_dk_2d.push_back(d_rho0_dk.at(d).getLocalView<Kokkos::HostSpace>());
    d_rho0_dk_1d.push_back(Kokkos::subview(d_rho0_dk_2d.at(d), Kokkos::ALL(), 0));
  }

  norm_d_rho0_dk.modify<Kokkos::HostSpace>();

  const LO num_local_elements = static_cast<LO>(kmb_map->getNodeNumElements());

  for (LO local_row = 0; local_row < num_local_elements; local_row++) {
    double norm = 0;
    for (std::size_t d = 0; d < k_dim; d++) {
      double component = d_rho0_dk_1d.at(d)(local_row);
      norm += component*component;
    }
    norm = std::sqrt(norm);
    norm_d_rho0_dk_1d(local_row) = norm;
  }

  norm_d_rho0_dk.sync<anomtrans::DistVecMemorySpace<double>>();

  std::cout << "made norm_d_rho0_dk" << std::endl;
  std::cout.flush();

  // norm_d_rho0_dk is distributed over all processes.
  // We need to move it to process 0 in order to write it out.
  // TODO are we sure this maintains the Map compatibility?
  // (i.e. do the global indices correspond?)
  const Tpetra::global_size_t num_global_indices = kmb.end_ikm;
  const std::size_t my_output_indices = (my_rank == 0) ? num_global_indices : 0;

  std::cout << "on rank " << my_rank << " num_global_indices = " << num_global_indices << "and my_output_indices = " << my_output_indices << std::endl;
  std::cout.flush();

  const GO index_base = 0;
  anomtrans::RCP<const anomtrans::Map> proc_zero_map(new anomtrans::Map(num_global_indices, my_output_indices, index_base, comm));

  std::cout << "made proc_zero_map" << std::endl;
  std::cout.flush();

  Tpetra::Export<LO, GO> exporter(kmb_map, proc_zero_map);

  std::cout << "made exporter" << std::endl;
  std::cout.flush();

  anomtrans::DistVec<double> out_norm_d_rho0_dk(proc_zero_map);
  out_norm_d_rho0_dk.doExport(norm_d_rho0_dk, exporter, Tpetra::REPLACE);

  std::cout << "exported norm_d_rho0_dk" << std::endl;
  std::cout.flush();

  if (my_rank == 0) {
    const LO out_num_local_elements = static_cast<LO>(proc_zero_map->getNodeNumElements());
    TEST_ASSERT(out_num_local_elements == num_global_indices );

    std::vector<double> out_stdvec_norm_d_rho0_dk(num_global_indices);
    std::vector<anomtrans::kComps<k_dim>> out_k_comps(num_global_indices);
    std::vector<unsigned int> out_ms(num_global_indices);

    out_norm_d_rho0_dk.sync<Kokkos::HostSpace>();
    auto out_norm_d_rho0_dk_2d = out_norm_d_rho0_dk.getLocalView<Kokkos::HostSpace>();
    auto out_norm_d_rho0_dk_1d = Kokkos::subview(out_norm_d_rho0_dk_2d, Kokkos::ALL(), 0);

    for (LO local_row = 0; local_row < out_num_local_elements; local_row++) {
      const GO global_row = proc_zero_map->getGlobalElement(local_row);
      auto ikm_comps = kmb.decompose(global_row);
      // nlohmann::json doesn't support std::tuple, so we need to break up ikm_comps.
      out_k_comps.at(global_row) = std::get<0>(ikm_comps);
      out_ms.at(global_row) = std::get<1>(ikm_comps);

      out_stdvec_norm_d_rho0_dk.at(global_row) = out_norm_d_rho0_dk_1d(local_row);
    }

    std::cout << "made output vectors" << std::endl;
    std::cout.flush();

    json j_out = {
      {"k_comps", out_k_comps},
      {"ms", out_ms},
      {"norm_d_rho0_dk", out_stdvec_norm_d_rho0_dk}
    };

    std::ofstream fp_out("rho0_test_out.json");
    fp_out << j_out.dump();
    fp_out.close();
    
    std::cout << "done" << std::endl;
    std::cout.flush();
    // TODO regression test: check that j_out hasn't changed compared to last run.
  }
}

} // namespace
