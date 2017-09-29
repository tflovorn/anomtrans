#ifndef ANOMTRANS_DISORDER_H
#define ANOMTRANS_DISORDER_H

#include <cmath>
#include <limits>
#include <complex>
#include <utility>
#include <tuple>
#include <petscksp.h>
#include "util/vec.h"
#include "util/constants.h"
#include "util/util.h"

namespace anomtrans {

/** @brief Calculate the disorder-averaged on-site diagonal disorder term
 *         which does not change band indices: <U_{km, k'm'} U_{k'm', km}>.
 *  @note There is an extra factor of U0^2/Nk appearing in <UU>. This is
 *        left out here to avoid passing the parameters. This function can be
 *        passed to make_collision() in a lambda which adds this factor.
 */
template <typename Hamiltonian>
double on_site_diagonal_disorder_band_preserved(const unsigned int Nbands,
    const Hamiltonian &H, const PetscInt ikm1, const PetscInt ikm2) {
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

/** @brief Calculate the disorder-averaged on-site diagonal disorder term
 *         which may change band indices: <U_{km, k'm'} U_{k'm', km''}>.
 *  @precondition Arguments are given such that k'' in ikm3 = (k'', m'') may
 *                be different than k in ikm1 = (k, m). For physically meaningful
 *                result (momentum conserved after disorder average), must have
 *                k'' = k. However, with the arguments passed here, cannot check this.
 *  @todo Add kmBasis to H (or pass kmBasis) to allow checking precondition.
 *  @note There is an extra factor of U0^2/Nk appearing in <UU>. This is
 *        left out here to avoid passing the parameters. This function can be
 *        passed to make_collision() in a lambda which adds this factor.
 */
template <typename Hamiltonian>
std::complex<double> on_site_diagonal_disorder(const unsigned int Nbands,
    const Hamiltonian &H, const PetscInt ikm1, const PetscInt ikm2, const PetscInt ikm3) {
  // Use Kahan summation for sum over band indices.
  std::complex<double> sum_i1(0.0, 0.0);
  std::complex<double> c_i1(0.0, 0.0);
  std::complex<double> sum_i2(0.0, 0.0);
  std::complex<double> c_i2(0.0, 0.0);

  // TODO check that k3 = k1.

  for (unsigned int i = 0; i < Nbands; i++) {
      std::complex<double> contrib_i1 = std::conj(H.basis_component(ikm1, i))
          * H.basis_component(ikm2, i);
      std::complex<double> contrib_i2 = std::conj(H.basis_component(ikm2, i))
          * H.basis_component(ikm3, i);

      std::complex<double> y = contrib_i1 - c_i1;
      std::complex<double> t = sum_i1 + y;
      c_i1 = (t - sum_i1) - y;
      sum_i1 = t;

      y = contrib_i2 - c_i2;
      t = sum_i2 + y;
      c_i2 = (t - sum_i2) - y;
      sum_i2 = t;
  }

  return sum_i1 * sum_i2;
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

  static kmBasis<k_dim> make_kb_diff(kmBasis<k_dim> kmb) {
    kComps<k_dim> Nk_diff;
    for (std::size_t d = 0; d < k_dim; d++) {
      Nk_diff.at(d) = 2*kmb.Nk.at(d) - 1;
    }
    kmBasis<k_dim> kb_diff(Nk_diff, 1);

    return kb_diff;
  }

  static std::vector<double> get_ULambda_vals(kmBasis<k_dim> kmb,
      DimMatrix<k_dim> D, double Lambda) {
    kmBasis<k_dim> kb_diff = make_kb_diff(kmb);

    std::vector<LatVec<k_dim>> L_lats;
    std::vector<double> L_Carts_norm2;
    std::tie(L_lats, L_Carts_norm2) = get_L_values(kmb, D, Lambda);
    assert(L_lats.size() == L_Carts_norm2.size());

    auto dks = get_dks(kmb, kb_diff);
    auto ULambda_elem = [D, Lambda, L_lats, L_Carts_norm2, dks](PetscInt ik_diff)->PetscScalar {
      auto dk = dks.at(ik_diff);
      return get_one_ULambda_val(D, Lambda, L_lats, L_Carts_norm2, dk);
    };

    Vec ULambda_vals_dist = vector_index_apply(kb_diff.end_ikm, ULambda_elem);

    VecScatter ctx;
    Vec ULambda_vals_all;
    PetscErrorCode ierr = VecScatterCreateToAll(ULambda_vals_dist, &ctx, &ULambda_vals_all);CHKERRXX(ierr);
    ierr = VecScatterBegin(ctx, ULambda_vals_dist, ULambda_vals_all, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);
    ierr = VecScatterEnd(ctx, ULambda_vals_dist, ULambda_vals_all, INSERT_VALUES, SCATTER_FORWARD);CHKERRXX(ierr);

    std::vector<PetscInt> all_rows;
    std::vector<PetscScalar> ULambda_vals_stdvec;
    std::tie(all_rows, ULambda_vals_stdvec) = get_local_contents(ULambda_vals_all);

    return split_scalars(ULambda_vals_stdvec).first;
  }

  /** @brief Collect all L values such that e^{-|L|^2/(2 Lambda^2)} > DBL_EPS.
   *  @todo Update to work with non-orthogonal basis.
   *        (Would permuting the order of iteration through d's be enough to
   *        include all possible L's?).
   */
  static std::pair<std::vector<LatVec<k_dim>>, std::vector<double>> get_L_values(kmBasis<k_dim> kmb,
      DimMatrix<k_dim> D, double Lambda) {
    // TODO: can we estimate how many L_lats we will collect and preallocate?
    std::vector<LatVec<k_dim>> L_lats;
    std::vector<double> L_Carts_norm2;
    LatVec<k_dim> zero;
    for (std::size_t d = 0; d < k_dim; d++) {
      zero.at(d) = 0.0;
    }
    L_lats.push_back(zero);
    L_Carts_norm2.push_back(0.0);

    std::vector<LatVec<k_dim>> L_lats_next;
    std::vector<double> L_Carts_norm2_next;
    for (std::size_t d = 0; d < k_dim; d++) {
      for (auto L_lat : L_lats) {
        L_lats_next.push_back(L_lat);
        L_Carts_norm2_next.push_back(norm2_CartVec(lat_vec_to_Cart(D, L_lat)));

        // Iterate through lattice vectors with increasing d component from L_lat.
        LatVec<k_dim> next_candidate = get_next_candidate(1, d, L_lat);
        double next_candidate_norm2 = norm2_CartVec(lat_vec_to_Cart(D, next_candidate));
        while (L_above_threshold(Lambda, next_candidate_norm2)) {
          L_lats_next.push_back(next_candidate);
          L_Carts_norm2_next.push_back(next_candidate_norm2);
          next_candidate = get_next_candidate(1, d, next_candidate);
          next_candidate_norm2 = norm2_CartVec(lat_vec_to_Cart(D, next_candidate));
        }

        // Iterate through lattice vectors with decreasing d component from L_lat.
        next_candidate = get_next_candidate(-1, d, L_lat);
        next_candidate_norm2 = norm2_CartVec(lat_vec_to_Cart(D, next_candidate));
        while (L_above_threshold(Lambda, next_candidate_norm2)) {
          L_lats_next.push_back(next_candidate);
          L_Carts_norm2_next.push_back(next_candidate_norm2);
          next_candidate = get_next_candidate(-1, d, next_candidate);
          next_candidate_norm2 = norm2_CartVec(lat_vec_to_Cart(D, next_candidate));
        }
      }
      L_lats = L_lats_next; // TODO sure this copy is adequate? Need explicit deep copy?
      L_lats_next.clear();
      L_Carts_norm2 = L_Carts_norm2_next;
      L_Carts_norm2_next.clear();
    }

    return std::make_pair(L_lats, L_Carts_norm2);
  }

  static LatVec<k_dim> get_next_candidate(int step, std::size_t d, LatVec<k_dim> L_lat) {
    LatVec<k_dim> next_candidate;
    for (std::size_t dp = 0; dp < k_dim; dp++) {
      if (dp == d) {
        next_candidate.at(dp) = L_lat.at(dp) + step;
      } else {
        next_candidate.at(dp) = L_lat.at(dp);
      }
    }
    return next_candidate;
  }

  static bool L_above_threshold(double Lambda, double norm2_L_Cart) {
    double fac = std::exp(-norm2_L_Cart / (2 * std::pow(Lambda, 2.0)));

    return fac > std::numeric_limits<double>::epsilon();
  }

  static std::vector<dkComps<k_dim>> get_dks(const kmBasis<k_dim> &kmb,
      const kmBasis<k_dim> kb_diff) {
    std::vector<dkComps<k_dim>> dks;
    dks.reserve(kb_diff.end_ikm);

    for (PetscInt ik_diff = 0; ik_diff < kb_diff.end_ikm; ik_diff++) {
      dkComps<k_dim> dk = ik_diff_to_dk(kmb, kb_diff, ik_diff);
      dks.push_back(dk);
    }

    return dks;
  }

  static dkComps<k_dim> ik_diff_to_dk(const kmBasis<k_dim> &kmb,
      const kmBasis<k_dim> &kb_diff, PetscInt ik_diff) {
    kmComps<k_dim> ik_diff_comps = kb_diff.decompose(ik_diff);
    dkComps<k_dim> dk;
    for (std::size_t d = 0; d < k_dim; d++) {
      dk.at(d) = std::get<0>(ik_diff_comps).at(d) - (kmb.Nk.at(d) - 1);
    }
    return dk;
  }

  static PetscInt dk_to_ik_diff(const kmBasis<k_dim> &kmb,
      const kmBasis<k_dim> &kb_diff, dkComps<k_dim> dk) {
    kmComps<k_dim> ik_diff_comps;
    for (std::size_t d = 0; d < k_dim; d++) {
      std::get<0>(ik_diff_comps).at(d) = dk.at(d) + kmb.Nk.at(d) - 1;
    }
    std::get<1>(ik_diff_comps) = 0;
    return kb_diff.compose(ik_diff_comps);
  }

  /** @brief Compute U_{Lambda}(k - k') for one value of k - k'.
   *  @todo Output must be real. By restricting L values to the subset with
   *        L_lats.at(0) >= 0 (and handling L_lats == 0 separately), can
   *        enforce this while also cutting number of L_lats in half.
   */
  static double get_one_ULambda_val(DimMatrix<k_dim> D, double Lambda,
      const std::vector<LatVec<k_dim>> &L_lats, const std::vector<double> &L_Carts_norm2,
      dkComps<k_dim> dk) {
    std::complex<double> sum(0.0, 0.0);
    std::complex<double> c(0.0, 0.0);
    // Sum over L's.
    for (std::size_t i = 0; i < L_lats.size(); i++) {
      LatVec<k_dim> L_lat = L_lats.at(i);
      double L_cart_norm2 = L_Carts_norm2.at(i);

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

  const std::vector<double> ULambda_vals;

public:
  const kmBasis<k_dim> kmb;
  const DimMatrix<k_dim> D;
  const double Lambda;
  const kmBasis<k_dim> kb_diff;

  SpatialDisorderCorrelation(kmBasis<k_dim> _kmb, DimMatrix<k_dim> _D, double _Lambda)
      : ULambda_vals(get_ULambda_vals(_kmb, _D, _Lambda)), kmb(_kmb), D(_D),
        Lambda(_Lambda), kb_diff(make_kb_diff(_kmb)) {}

  /** @brief Calculate the band-independent spatial correlation factor in the disorder term.
   */
  double operator()(const PetscInt ikm1, const PetscInt ikm2) const {
    kmComps<k_dim> km1 = kmb.decompose(ikm1);
    kmComps<k_dim> km2 = kmb.decompose(ikm2);
    dkComps<k_dim> dk;
    for (std::size_t d = 0; d < k_dim; d++) {
      dk.at(d) = std::get<0>(km1).at(d) - std::get<0>(km2).at(d);
    }
    PetscInt ik_diff = dk_to_ik_diff(kmb, kb_diff, dk);
    return ULambda_vals.at(ik_diff);
  }
};

/** @brief Calculate the disorder-averaged on-site diagonal disorder term
 *         which does not change band indices: <U_{km, k'm'} U_{k'm', km}>.
 *  @note There is an extra factor of U0^2/Nk appearing in <UU>. This is
 *        left out here to avoid passing the parameters. This function can be
 *        passed to make_collision() in a lambda which adds this factor.
 */
template <typename Hamiltonian, typename spatial_correlation>
double spatially_correlated_diagonal_disorder_band_preserved(const unsigned int Nbands,
    const Hamiltonian &H, const spatial_correlation &ULambda,
    const PetscInt ikm1, const PetscInt ikm2) {
  return on_site_diagonal_disorder_band_preserved(Nbands, H, ikm1, ikm2) * ULambda(ikm1, ikm2);
}

/** @brief Calculate the disorder-averaged on-site diagonal disorder term
 *         which may change band indices: <U_{km, k'm'} U_{k'm', km''}>.
 *  @note There is an extra factor of U0^2/Nk appearing in <UU>. This is
 *        left out here to avoid passing the parameters. This function can be
 *        passed to make_collision() in a lambda which adds this factor.
 */
template <typename Hamiltonian, typename spatial_correlation>
std::complex<double> spatially_correlated_diagonal_disorder(const unsigned int Nbands,
    const Hamiltonian &H, const spatial_correlation &ULambda,
    const PetscInt ikm1, const PetscInt ikm2, const PetscInt ikm3) {
  return on_site_diagonal_disorder(Nbands, H, ikm1, ikm2, ikm3) * ULambda(ikm1, ikm2);
}

} // namespace anomtrans

#endif // ANOMTRANS_DISORDER_H
