#ifndef ANOMTRANS_CONDUCTIVITY_H
#define ANOMTRANS_CONDUCTIVITY_H

#include <stdexcept>
#include <tuple>
#include <petscksp.h>
#include "util/vec.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief Compute \hbar times <v>, the expectation value of the velocity operator:
 *         \hbar <v> = \hbar Tr[v <rho>] = \sum_{kmm'} <km|\grad_k H_k|km'> <rho>_k^{mm'}.
 *  @returns The Cartesian components of \hbar <v>.
 *  @todo Return PetscReal instead of PetscScalar? Output should be guaranteed to be real.
 *  @todo Implement using helper function that constructs [v]_{km, km'} matrix.
 *        Allow for use with spin current.
 *  @todo Add (trivial) function to calculate current (-v in e/hbar units).
 */
template <std::size_t k_dim, typename Hamiltonian>
std::array<PetscScalar, k_dim> calculate_velocity_ev(const kmBasis<k_dim> &kmb,
    const Hamiltonian &H, Mat rho) {
  auto row_elem = [&kmb, &H, rho](PetscInt ikmp)->std::array<PetscScalar, k_dim> {
    auto mp = std::get<1>(kmb.decompose(ikmp));

    std::array<PetscScalar, k_dim> row_total, c;
    // TODO can omit initialization of row_total, c here?
    // Appears that std::complex<double> default-initializes to 0.
    for (std::size_t dc = 0; dc < k_dim; dc++) {
      row_total.at(dc) = 0.0;
      c.at(dc) = 0.0;
    }

    PetscInt ncols;
    const PetscInt *cols;
    const PetscScalar *rho_row;
    PetscErrorCode ierr = MatGetRow(rho, ikmp, &ncols, &cols, &rho_row);CHKERRXX(ierr);

    for (PetscInt col_index = 0; col_index < ncols; col_index++) {
      PetscInt ikm = cols[col_index];
      auto ikm_comps = kmb.decompose(ikm);

      auto grad_k_mmp = H.gradient(ikm_comps, mp);
      PetscScalar rho_k_mpm = rho_row[col_index];

      // Use Kahan sum to avoid introducing error if the number of bands is large.
      // TODO - prefer a different summation strategy?
      for (std::size_t dc = 0; dc < k_dim; dc++) {
        PetscScalar contrib = grad_k_mmp.at(dc) * rho_k_mpm;

        PetscScalar y = contrib - c.at(dc);
        PetscScalar t = row_total.at(dc) + y;
        c.at(dc) = (t - row_total.at(dc)) - y;
        row_total.at(dc) = t;
      }
    }

    ierr = MatRestoreRow(rho, ikmp, &ncols, &cols, &rho_row);CHKERRXX(ierr);

    return row_total;
  };

  std::array<Vec, k_dim> v_vecs = vector_index_apply_multiple<k_dim>(kmb.end_ikm, row_elem);

  std::array<PetscScalar, k_dim> vs;
  for (std::size_t dc = 0; dc < k_dim; dc++) {
    PetscScalar component;
    PetscErrorCode ierr = VecSum(v_vecs.at(dc), &component);CHKERRXX(ierr);

    vs.at(dc) = component;

    ierr = VecDestroy(&(v_vecs.at(dc)));CHKERRXX(ierr);
  }

  return vs;
}

} // namespace anomtrans

#endif // ANOMTRANS_CONDUCTIVITY_H
