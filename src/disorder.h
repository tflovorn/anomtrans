#ifndef ANOMTRANS_DISORDER_H
#define ANOMTRANS_DISORDER_H

#include <cmath>
#include <limits>
#include <complex>

namespace anomtrans {

/** @brief Calculate the disorder-averaged on-site diagonal disorder term.
 *  @note There is an extra factor of U0^2/Nk appearing in <UU>. This is
 *        left out here to avoid passing the parameters.
 */
template <typename Hamiltonian>
double on_site_diagonal_disorder(const unsigned int Nbands, const Hamiltonian &H,
    const PetscInt ikm1, const PetscInt ikm2) {
  // Use Kahan summation for sum over band indices.
  std::complex<double> sum(0.0, 0.0);
  std::complex<double> c(0.0, 0.0);
  for (unsigned int i = 0; i < Nbands; i++) {
      std::complex<double> contrib = std::conj(H.basis_component(ikm1, i))
          * H.basis_component(ikm2, i);

      std::complex<double> y = contrib - c;
      std::complex<double> t = sum + y;
      c = (t - sum) - y;
      sum = t;
  }

  return std::norm(sum);
}

} // namespace anomtrans

#endif // ANOMTRANS_DISORDER_H
