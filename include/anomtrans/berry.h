#ifndef ANOMTRANS_BERRY_H
#define ANOMTRANS_BERRY_H

#include <cstddef>
#include <array>
#include <vector>
#include <tuple>
#include <petscksp.h>
#include "util/mat.h"
#include "grid_basis.h"
#include "util/util.h"

namespace anomtrans {

/** @brief Calculate the Berry connection for the given Hamiltonian.
 *  @note Degeneracies are treated by introducing a Lorentzian broadening:
 *        1 / (E_{km'} - E_{km}) is replaced by
 *        (E_{km'} - E_{km}) / ((E_{km'} - E_{km})^2 + broadening^2).
 */
template <std::size_t k_dim, typename Hamiltonian>
std::array<Mat, k_dim> make_berry_connection(const kmBasis<k_dim> &kmb, const Hamiltonian &H,
    double broadening) {
  static_assert(k_dim > 0, "must have number of spatial dimensions > 0");

  std::array<Mat, k_dim> result;
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    result.at(dc) = make_Mat(kmb.end_ikm, kmb.end_ikm, kmb.Nbands);
  }

  PetscInt begin, end;
  PetscErrorCode ierr = MatGetOwnershipRange(result.at(0), &begin, &end);CHKERRXX(ierr);

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    std::vector<PetscInt> result_row_cols;
    result_row_cols.reserve(kmb.Nbands);

    std::array<std::vector<std::complex<double>>, k_dim> result_row_vals;
    for (std::size_t dc = 0; dc < k_dim; dc++) {
      result_row_vals.at(dc).reserve(kmb.Nbands);
    }

    kmComps<k_dim> km = kmb.decompose(local_row);
    unsigned int m = std::get<1>(km);

    for (unsigned int mp = 0; mp < kmb.Nbands; mp++) {
      if (mp == m) {
        continue;
      }

      kmComps<k_dim> kmp = std::make_tuple(std::get<0>(km), mp);

      double ediff = H.energy(kmp) - H.energy(km);
      std::complex<double> coeff(0.0, ediff / (std::pow(ediff, 2.0) + std::pow(broadening, 2.0)));
      auto grad = H.gradient(km, mp);

      result_row_cols.push_back(kmb.compose(kmp));

      for (std::size_t dc = 0; dc < k_dim; dc++) {
        std::complex<double> val = coeff * grad.at(dc);
        result_row_vals.at(dc).push_back(val);
      }
    }

    for (std::size_t dc = 0; dc < k_dim; dc++) {
      PetscErrorCode ierr = MatSetValues(result.at(dc), 1, &local_row, result_row_cols.size(),
          result_row_cols.data(), result_row_vals.at(dc).data(), INSERT_VALUES);CHKERRXX(ierr);
    }
  }

  for (std::size_t dc = 0; dc < k_dim; dc++) {
    PetscErrorCode ierr = MatAssemblyBegin(result.at(dc), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
    ierr = MatAssemblyEnd(result.at(dc), MAT_FINAL_ASSEMBLY);CHKERRXX(ierr);
  }

  return result;
}

/** @brief Calculate the Berry curvature for the given Hamiltonian.
 *  @note Degeneracies are treated by introducing a Lorentzian broadening:
 *        1 / (E_{km'} - E_{km})^2 is replaced by
 *        1 / ((E_{km'} - E_{km})^2 + broadening^2).
 */
template <std::size_t k_dim, typename Hamiltonian>
std::array<Vec, 3> make_berry_curvature(const kmBasis<k_dim> &kmb, const Hamiltonian &H,
    double broadening) {
  static_assert(k_dim > 0, "must have number of spatial dimensions > 0");

  std::array<Vec, 3> result;

  for (std::size_t dc = 0; dc < 3; dc++) {
    PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm,
        &(result.at(dc)));CHKERRXX(ierr);
  }

  PetscInt begin, end;
  PetscErrorCode ierr = VecGetOwnershipRange(result.at(0), &begin, &end);CHKERRXX(ierr);

  std::vector<PetscInt> local_rows;
  std::array<std::vector<PetscScalar>, 3> local_vals;

  local_rows.reserve(end - begin);
  for (std::size_t dc = 0; dc < 3; dc++) {
    local_vals.at(dc).reserve(end - begin);
  }

  for (PetscInt local_row = begin; local_row < end; local_row++) {
    kmComps<k_dim> km = kmb.decompose(local_row);
    unsigned int m = std::get<1>(km);

    local_rows.push_back(local_row);
    for (std::size_t dc = 0; dc < 3; dc++) {
      local_vals.at(dc).push_back(std::complex<double>(0.0, 0.0));
    }

    for (unsigned int mp = 0; mp < kmb.Nbands; mp++) {
      if (mp == m) {
        continue;
      }

      kmComps<k_dim> kmp = std::make_tuple(std::get<0>(km), mp);

      double ediff = H.energy(km) - H.energy(kmp);
      double denom = std::pow(ediff, 2.0) + std::pow(broadening, 2.0);

      auto grad = H.gradient(km, mp);
      auto grad_star = H.gradient(kmp, m);

      auto grad_cross = cross(grad, grad_star);

      for (std::size_t dc = 0; dc < 3; dc++) {
        // [i (a x a^*)] is pure real for all a. Make sure Berry curvature is real.
        // TODO - is there a good general way to choose scale of absolute error here?
        assert(scalars_approx_equal(grad_cross.at(dc), -std::conj(grad_cross.at(dc)),
              10.0*std::numeric_limits<PetscReal>::epsilon(),
              10.0*std::numeric_limits<PetscReal>::epsilon()));

        PetscScalar num = std::complex<double>(0.0, 1.0) * grad_cross.at(dc);
        // TODO prefer Kahan sum here?
        // Error ~ Nbands.
        local_vals.at(dc).at(local_row - begin) += num / denom;
      }
    }
  }

  for (std::size_t dc = 0; dc < 3; dc++) {
    assert(local_rows.size() == local_vals.at(dc).size());

    ierr = VecSetValues(result.at(dc), local_rows.size(), local_rows.data(),
        local_vals.at(dc).data(), INSERT_VALUES);CHKERRXX(ierr);

    ierr = VecAssemblyBegin(result.at(dc));CHKERRXX(ierr);
    ierr = VecAssemblyEnd(result.at(dc));CHKERRXX(ierr);
  }

  return result;
}

} // namespace anomtrans

#endif // ANOMTRANS_BERRY_H
