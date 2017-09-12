#ifndef ANOMTRANS_HAMILTONIAN_H
#define ANOMTRANS_HAMILTONIAN_H

#include <cstddef>
#include <complex>
#include <array>
#include <vector>
#include <petscksp.h>
#include "util.h"
#include "mat.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief Calculate [DH_0/Dk] x \hat{B}.
 *  @note `DH0_cross_Bhat` is used only in `apply_driving_magnetic`, where the dot product
 *        of it with D<rho>/Dk is taken. Since D<rho>/Dk has finite components only
 *        for 0 <= dc < k_dim, only components with dc in this range are given here.
 */
template <std::size_t k_dim, typename Hamiltonian>
std::array<Mat, k_dim> make_DH0_cross_Bhat(const kmBasis<k_dim> &kmb, const Hamiltonian &H,
    std::array<double, 3> Bhat) {
  std::array<Mat, k_dim> result;
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    result.at(dc) = make_Mat(kmb.end_ikm, kmb.end_ikm, kmb.Nbands);
  }

  for (PetscInt ikm = 0; ikm < kmb.end_ikm; ikm++) {
    std::vector<PetscInt> result_row_cols;
    result_row_cols.reserve(kmb.Nbands);

    std::array<std::vector<std::complex<double>>, k_dim> result_row_vals;
    for (std::size_t dc = 0; dc < k_dim; dc++) {
      result_row_vals.at(dc).reserve(kmb.Nbands);
    }

    kmComps<k_dim> km = kmb.decompose(ikm);

    for (unsigned int mp = 0; mp < kmb.Nbands; mp++) {
      auto grad = H.gradient(km, mp);
      auto grad_cross_B = cross(grad, make_complex_array(Bhat));

      PetscInt ikmp = kmb.compose(make_tuple(std::get<0>(km), mp));
      result_row_cols.push_back(ikmp);

      for (std::size_t dc = 0; dc < k_dim; dc++) {
        result_row_vals.at(dc).push_back(grad_cross_B.at(dc));
      }
    }

    for (std::size_t dc = 0; dc < k_dim; dc++) {
      PetscErrorCode ierr = MatSetValues(result.at(dc), 1, &ikm, result_row_cols.size(),
          result_row_cols.data(), result_row_vals.at(dc).data(), INSERT_VALUES);CHKERRXX(ierr);
    }
  }

  for (std::size_t dc = 0; dc < k_dim; dc++) {
    PetscErrorCode ierr = MatAssemblyBegin(result.at(dc), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    ierr = MatAssemblyEnd(result.at(dc), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  }

  return result;
}

} // namespace anomtrans

#endif // ANOMTRANS_HAMILTONIAN_H
