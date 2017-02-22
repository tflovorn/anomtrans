#ifndef ANOMTRANS_DISORDER_H
#define ANOMTRANS_DISORDER_H

#include <cmath>
#include <limits>
#include <complex>
#include <utility>
#include <tuple>
#include <map>
#include "constants.h"

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

/** @brief Functor which computes the spatially correlated disorder factor
 *         U_{Lambda}(k - k') = \sum_L e^{i (k - k') dot L} e^{-|L|^2/(2 Lambda^2)}.
 *         On construction of an instance of this object, this sum is performed
 *         for all k - k' values and cached.
 *  @todo Implement correct logic for exhausting all L's for nonorthogonal basis
 *        vectors in get_ULambda_vals.
 *  @todo Is is worth parallelizing the construction of this object? Number of
 *        k - k' values scales linearly with number of k values. Can ULambda_vals
 *        be represented as a PETSc Vec?
 */
template <std::size_t k_dim>
class SpatialDisorderCorrelation {
  // This class doesn't make sense for k_dim = 0.
  static_assert(k_dim > 0, "k must have at least one component");

  using LLat = std::array<int, k_dim>;

  static std::map<dkComps<k_dim>, double> get_ULambda_vals(kmBasis<k_dim> kmb,
      DimMatrix<k_dim> D, double Lambda) {
    std::vector<LLat> L_lats;
    std::vector<double> L_carts_norm2;
    std::tie(L_lats, L_carts_norm2) = get_L_values(kmb, D, Lambda);
    assert(L_lats.size() == L_carts_norm2.size());

    std::map<dkComps<k_dim>, double> ULambda_vals_collect;

    auto dks = get_dks(kmb);
    for (auto dk : dks) {
      ULambda_vals_collect[dk] = get_one_ULambda_val(D, Lambda, L_lats,
          L_carts_norm2, dk);
    }

    return ULambda_vals_collect;
  }

  /** @brief Collect all L values such that e^{-|L|^2/(2 Lambda^2)} > DBL_EPS.
   *  @todo Update to work with non-orthogonal basis.
   *        (Would permuting the order of iteration through d's be enough to
   *        include all possible L's?).
   */
  static std::pair<std::vector<LLat>, std::vector<double>> get_L_values(kmBasis<k_dim> kmb,
      DimMatrix<k_dim> D, double Lambda) {
    // TODO: can we estimate how many L_lats we will collect and preallocate?
    std::vector<LLat> L_lats;
    LLat zero;
    for (std::size_t d = 0; d < k_dim; d++) {
      zero.at(d) = 0.0;
    }
    L_lats.append(zero);

    std::vector<LLat> L_lats_next;
    for (std::size_t d = 0; d < k_dim; d++) {
      for (auto L_lat : L_lats) {
        L_lats_next.push_back(L_lat);

        LLat next_candidate = get_next_candidate(1, d, L_lat);
        while (L_above_threshold(D, Lambda, next_candidate)) {
          L_lats_next.push_back(next_candidate);
          next_candidate = get_next_candidate(1, d, next_candidate);
        }

        next_candidate = get_next_candidate(-1, d, L_lat);
        while (L_above_threshold(D, Lambda, next_candidate)) {
          L_lats_next.push_back(next_candidate);
          next_candidate = get_next_candidate(-1, d, next_candidate);
        }
      }
      L_lats = L_lats_next;
    }
    // TODO could generate L_carts_norm2 while generating L_lats (need to do
    // L_lat -> L_cart to check threshold).

    return L_lats;
  }

  static LLat get_next_candidate(int step, std::size_t d, LLat L_lat) {
    LLat next_candidate;
    for (std::size_t dp = 0; dp < k_dim; dp++) {
      if (dp == d) {
        next_candidate.at(dp) = L_lat.at(d) + step;
      } else {
        next_candidate.ad(dp) = L_lat.at(d);
      }
    }
  }

  static bool L_above_threshold(DimMatrix<k_dim> D, double Lambda, LLat L_lat) {
    std::array<double, k_dim> L_Cart;
    // TODO could factor out L_lat -> L_Cart
    for (std::size_t dc = 0; dc < k_dim; dc++) {
      L_Cart.at(dc) = 0.0;
      for (std::size_t di = 0; di < k_dim; di++) {
        L_Cart.at(dc) += L_lat.at(di) * D.at(dc).at(di);
      }
    }

    double norm2_L_Cart = 0.0;
    for (std::size_t dc = 0; dc < k_dim; dc++) {
      norm2_L_Cart += std::pow(L_Cart.at(dc), 2.0);
    }

    double fac = std::exp(-norm2_L_Cart / (2 * std::pow(Lambda, 2.0)));

    return fac > std::numeric_limits<double>::epsilon();
  }

  static std::vector<dkComps<k_dim>> get_dks(kmBasis<k_dim> kmb) {
    kComps<k_dim> Nk_diff;
    for (std::size_t d = 0; d < k_dim; d++) {
      Nk_diff.at(d) = 2*kmb.Nk.at(d) - 1;
    }
    kmBasis<k_dim> kb_diff(Nk_diff, 1);

    std::vector<dkComps<k_dim>> dks;
    dks.reserve(kb_diff.end_ikm);

    for (PetscInt ik_diff = 0; ik_diff < kb_diff.end_ikm; ik_diff++) {
      kmComps<k_dim> ik_diff_comps = kb_diff.decompose(ik_diff);
      dkComps<k_dim> dk;
      for (std::size_t d = 0; d < k_dim; d++) {
        dk.at(d) = std::get<0>(ik_diff_comps).at(d) - (kmb.Nk.at(d) - 1);
      }
      dks.push_back(dk);
    }

    return dks;
  }

  /** @brief Compute U_{Lambda}(k - k') for one value of k - k'.
   *  @todo Output must be real. By restricting L values to the subset with
   *        L_lats.at(0) >= 0 (and handling L_lats == 0 separately), can
   *        enforce this while also cutting number of L_lats in half.
   */
  static double get_one_ULambda_val(DimMatrix<k_dim> D, double Lambda,
      std::vector<LLat> L_lats, std::vector<double> L_carts_norm2,
      dkComps<k_dim> dk) {
    std::complex<double> sum(0.0, 0.0);
    std::complex<double> c(0.0, 0.0);
    // Sum over L's.
    for (std::size_t i = 0; i < L_lats.size(); i++) {
      LLat L_lat = L_lats.at(i);
      double L_cart_norm2 = L_carts_norm2.at(i);

      double dk_dot_L = 0.0;
      for (std::size_t d = 0; d < k_dim; d++) {
        dk_dot_L += 2*pi*dk.at(d)*L_lat.at(d);
      }
      // e^{i dk dot L} e^{-|L|^2/(2 Lambda^2)}
      std::complex<double> contrib = std::exp(std::complex<double>(0.0, dk_dot_L))
        * std::exp(-L_cart_norm2/(2*std::pow(Lambda, 2.0)));

      std::complex<double> y = contrib - c;
      std::complex<double> t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }

    assert(sum.imag() < std::numeric_limits<double>::epsilon());

    return sum.real();
  }

  const std::map<dkComps<k_dim>, double> ULambda_vals;

public:
  const kmBasis<k_dim> kmb;
  const DimMatrix<k_dim> D;
  const double Lambda;

  SpatialDisorderCorrelation(kmBasis<k_dim> _kmb, DimMatrix<k_dim> _D, double _Lambda)
      : kmb(_kmb), ULambda_vals(get_ULambda_vals(_kmb, _D, _Lambda)) {}

  double operator()(PetscInt ikm1, PetscInt ikm2) {
    kmComps<k_dim> km1 = kmb.decompose(ikm1);
    kmComps<k_dim> km2 = kmb.decompose(ikm2);
    dkComps<k_dim> dk;
    for (std::size_t d = 0; d < k_dim; d++) {
      dk.at(d) = std::get<0>(km1).at(d) - std::get<0>(km2).at(d);
    }
    return ULambda_vals.at(dk);
  }
};

template <typename Hamiltonian, typename spatial_correlation>
double spatially_correlated_diagonal_disorder(const unsigned int Nbands,
    const Hamiltonian &H, const spatial_correlation ULambda,
    const PetscInt ikm1, const PetscInt ikm2) {
  return on_site_diagonal_disorder(Nbands, H, ikm1, ikm2) * ULambda(ikm1, ikm2);
}

} // namespace anomtrans

#endif // ANOMTRANS_DISORDER_H
