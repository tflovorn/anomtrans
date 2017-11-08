#ifndef ANOMTRANS_WANNIER90_HAMILTONIAN_H
#define ANOMTRANS_WANNIER90_HAMILTONIAN_H

#include <cstddef>
#include <complex>
#include <exception>
#include <array>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <petscksp.h>
#include "util/constants.h"
#include "util/lattice.h"
#include "grid_basis.h"
#include "observables/spin.h"

namespace anomtrans {

/** @brief Returns the full spin matrices (S_x, S_y, S_z) (in units of hbar) for
 *         a system in which the basis alternates spin:
 *         ((orbital 0, up), (orbital 0, down), (orbital 1, up), (orbital 1, down), ...)
 */
std::array<Eigen::MatrixXcd, 3> full_spin_matrices(std::size_t Nbands);

/** @brief Tight-binding Hamiltonian, represented by a set of hopping matrices
 *         Hrs[r](ip, i) giving the matrix elements <ip, 0|H|i, r>.
 *         Assumes the system has both spins present and is given in a basis of
 *         alternating spin:
 *         ((orbital 0, up), (orbital 0, down), (orbital 1, up), (orbital 1, down), ...)
 *  @todo Impose maximum size on caches.
 */
template <std::size_t k_dim>
class TBHamiltonian {
  /** @brief Calculate H(k) by summing the Fourier series.
   */
  Eigen::MatrixXcd calc_Hk_at(kComps<k_dim> k) const {
    kVals<k_dim> kv = std::get<0>(km_at(kmb.Nk, std::make_tuple(k, 0u)));

    Eigen::MatrixXcd Hk = Eigen::MatrixXcd::Zero(kmb.Nbands, kmb.Nbands);

    for (auto it = Hrs.begin(); it != Hrs.end(); ++it) {
      LatVec<k_dim> r = it->first;
      double k_dot_r = 0.0;
      for (std::size_t di = 0; di < k_dim; di++) {
        k_dot_r += 2.0 * pi * kv.at(di) * r.at(di);
      }
      std::complex<double> coeff = std::exp(std::complex<double>(0.0, k_dot_r));

      Hk += coeff * it->second;
    }

    return Hk;
  }

  /** @brief Calculate grad_k H(k) using the analytic differentiation of the
   *         Fourier series yielding H(k).
   */
  std::array<Eigen::MatrixXcd, k_dim> calc_grad_Hk_at(kComps<k_dim> k) const {
    kVals<k_dim> kv = km_at(k, kmb.Nk);

    std::array<Eigen::MatrixXcd, k_dim> grad_Hk;
    for (std::size_t dc = 0; dc < k_dim; dc++) {
      grad_Hk.at(dc) = Eigen::MatrixXcd::Zero(kmb.Nbands, kmb.Nbands);
    }

    for (auto it = Hrs.begin(); it != Hrs.end(); ++it) {
      LatVec<k_dim> r = it->first;
      CartVec<k_dim> r_Cart = lat_vec_to_Cart(D, r);

      double k_dot_r = 0.0;
      for (std::size_t di = 0; di < k_dim; di++) {
        k_dot_r += 2.0 * pi * kv.at(di) * r.at(di);
      }
      std::complex<double> exp_coeff = std::exp(std::complex<double>(0.0, k_dot_r));

			for (std::size_t dc = 0; dc < k_dim; dc++) {
				std::complex<double> coeff = std::complex<double>(0.0, r_Cart.at(dc)) * exp_coeff;

				grad_Hk.at(dc) += coeff * it->second;
			}
    }

    return grad_Hk;
  }

  const Eigen::VectorXd& Ek_at(kComps<k_dim> k) const {
    auto Ek_prev = Ek_cache.find(k);
    if (Ek_prev != Ek_cache.end()) {
      return Ek_prev->second;
    }

    auto Hk = calc_Hk_at(k);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigensolver(Hk);
    if (eigensolver.info() != Eigen::Success) {
      throw std::runtime_error("eigensolver failed");
    }

    Ek_cache[k] = eigensolver.eigenvalues();
    Uk_cache[k] = eigensolver.eigenvectors();

    return Ek_cache[k];
  }

  const Eigen::MatrixXcd& Uk_at(kComps<k_dim> k) const {
    auto Uk_prev = Uk_cache.find(k);
    if (Uk_prev != Uk_cache.end()) {
      return Uk_prev->second;
    }

    // TODO same logic as Ek_at; consolidate.
    auto Hk = calc_Hk_at(k);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigensolver(Hk);
    if (eigensolver.info() != Eigen::Success) {
      throw std::runtime_error("eigensolver failed");
    }

    Ek_cache[k] = eigensolver.eigenvalues();
    Uk_cache[k] = eigensolver.eigenvectors();

    return Uk_cache[k];
  }

  const std::array<Eigen::MatrixXcd, k_dim>& grad_at(kComps<k_dim> k) const {
    auto grad_prev = grad_Hk_eigenbasis_cache.find(k);
    if (grad_prev != grad_Hk_eigenbasis_cache.end()) {
      return grad_prev->second;
    }

    const auto& Uk = Uk_at(k);
    std::array<Eigen::MatrixXcd, k_dim> grad_Hk = calc_grad_Hk_at(k);

    std::array<Eigen::MatrixXcd, k_dim> grad_Hk_eigenbasis;
    for (std::size_t dc = 0; dc < k_dim; dc++) {
      grad_Hk_eigenbasis.at(dc) = Uk.adjoint() * grad_Hk.at(dc) * Uk;
    }

    grad_Hk_eigenbasis_cache[k] = grad_Hk_eigenbasis;
    return grad_Hk_eigenbasis_cache[k];
  }

  const std::array<Eigen::MatrixXcd, 3>& spin_at(kComps<k_dim> k) const {
    auto spin_prev = spin_eigenbasis_cache.find(k);
    if (spin_prev != spin_eigenbasis_cache.end()) {
      return spin_prev->second;
    }

    const auto& Uk = Uk_at(k);
    std::array<Eigen::MatrixXcd, 3> spin_eigenbasis;
    for (std::size_t dc = 0; dc < 3; dc++) {
      spin_eigenbasis.at(dc) = Uk.adjoint() * spin.at(dc) * Uk;
    }

    spin_eigenbasis_cache[k] = spin_eigenbasis;
    return spin_eigenbasis_cache[k];
  }

  mutable std::map<kComps<k_dim>, Eigen::VectorXd> Ek_cache;
  mutable std::map<kComps<k_dim>, Eigen::MatrixXcd> Uk_cache;
  mutable std::map<kComps<k_dim>, std::array<Eigen::MatrixXcd, k_dim>> grad_Hk_eigenbasis_cache;
  mutable std::map<kComps<k_dim>, std::array<Eigen::MatrixXcd, 3>> spin_eigenbasis_cache;

  const std::array<Eigen::MatrixXcd, 3> spin_operator;

public:
  /** @brief Tight-binding Hamiltonian matrix. 
   */
  const std::map<LatVec<k_dim>, Eigen::MatrixXcd> Hrs;

  /** @brief Discretization of (k, m) basis to use for Fourier transform.
   */
  const kmBasis<k_dim> kmb;

  /** @brief A matrix giving the lattice vectors: D[c][i] is the c'th Cartesian
   *         component of the i'th lattice vector.
	 */
	const DimMatrix<k_dim> D;

  TBHamiltonian(std::map<LatVec<k_dim>, Eigen::MatrixXcd> _Hrs, kComps<k_dim> _Nk,
      std::size_t _Nbands, DimMatrix<k_dim> _D) : spin_operator(full_spin_matrices(_Nbands)), Hrs(_Hrs),
      kmb(kmBasis<k_dim>(_Nk, _Nbands)), D(_D) {}

  /** @brief Energy at (k,m): E_{km}.
   */
  double energy(kmComps<k_dim> ikm_comps) const {
    kComps<k_dim> k;
    unsigned int m;
    std::tie(k, m) = ikm_comps;

    return Ek_at(k)(m);
  }

  /** @brief Value of U_{im}(k), where U is the unitary matrix which diagonalizes
   *         H(k), m is the eigenvalue index, and i is the component of the
   *         initial basis (pseudo-atomic orbital or otherwise).
   */
  std::complex<double> basis_component(PetscInt ikm, unsigned int i) const {
    kComps<k_dim> k;
    unsigned int m;
    std::tie(k, m) = kmb.decompose(ikm);

    return Uk_at(k)(m, i);
  }

  /** @brief Gradient of the Hamiltonian, evaluated in the eigenbasis;
   *         equal to the covariant derivative of the Hamiltonain.
   *         gradient(ikm, mp) = <k, m|grad_k H|k, mp>.
   */
  std::array<std::complex<double>, k_dim> gradient(kmComps<k_dim> ikm_comps, unsigned int mp) const {
    kComps<k_dim> k;
    unsigned int m;
    std::tie(k, m) = ikm_comps;

    const auto& grad = grad_at(k);
    std::array<std::complex<double>, k_dim> grad_val;
    for (std::size_t dc = 0; dc < k_dim; dc++) {
      grad_val.at(dc) = grad.at(dc)(m, mp);
    }
    return grad_val;
  }

  /** @brief Spin, evaluated in the eigenbasis (units of hbar):
   *         spin(ikm, mp)[a] = <km|S_a|km'>
   */
  std::array<std::complex<double>, 3> spin(PetscInt ikm, unsigned int mp) const {
    kComps<k_dim> k;
    unsigned int m;
    std::tie(k, m) = kmb.decompose(ikm);

    const auto& spin = spin_at(k);
    std::array<std::complex<double>, 3> spin_val;
    for (std::size_t dc = 0; dc < 3; dc++) {
      spin_val.at(dc) = spin.at(dc)(m, mp);
    }
    return spin_val;
  }
};

struct HrHeader {
  std::size_t Nbands, Nrs;
  std::vector<unsigned int> degen;
  std::size_t start_hr;
};

/** @brief Parse the header lines from the hr.dat file at hr_path.
 *
 *  @note The Wannier90 hr.dat file header has the format:
 *
 *        comment line
 *        number of bands
 *        number of displacement vectors (rs)
 *        list of degen values, 15 per line, total number equal to rs
 */
HrHeader extract_hr_header(std::string hr_path);

namespace internal {

/** @brief Convert (ra, rb, rc) to the appropriate LatVec<k_dim>.
 *         Discard unused r values.
 *  @example process_LatVec<2>(ra, rb, rc) = {ra, rb}.
 *  @todo Implementing this using template specialization. Would prefer to use constexpr if.
 */
template <std::size_t k_dim>
LatVec<k_dim> process_LatVec(int ra, int rb, int rc);

template <>
LatVec<1> process_LatVec(int ra, int rb, int rc);

template <>
LatVec<2> process_LatVec(int ra, int rb, int rc);

template <>
LatVec<3> process_LatVec(int ra, int rb, int rc);

}

/** @brief Parse the tight-binding Hamiltonian from the hr.dat file at hr_path.
 *
 *  @note The tight-binding Hamiltonian is specified in the Wannier90 hr.dat file
 *        one matrix element per line, with each line having the format:
 *
 *        Ra  Rb  Rc  ip  i  Re{[H(R)]_{ip, i}}*degen[R]  Im{[H(R)]_{ip, i}}*degen[R]
 *
 *        ip and i are tight-binding basis indices, and R is the displacement vector.
 *        ip varies the fastest, then i, then R.
 *        ip and i are given in the file starting at 1, not 0. We store them here
 *        starting at 0.
 */
template <std::size_t k_dim>
std::map<LatVec<k_dim>, Eigen::MatrixXcd> extract_hr_model(std::string hr_path,
    const HrHeader &header) {
  std::ifstream fp(hr_path);
  std::string line;
  // Skip the header.
  for (std::size_t line_number = 0; line_number < header.start_hr; line_number++) {
    std::getline(fp, line);
  }

  // Extract the tight-binding matrix elements.
  std::map<LatVec<k_dim>, Eigen::MatrixXcd> H_tb;
  for (std::size_t r_index = 0; r_index < header.Nrs; r_index++) {
    Eigen::MatrixXcd Hr(header.Nbands, header.Nbands);
    LatVec<k_dim> r;

    for (std::size_t i = 0; i < header.Nbands; i++) {
      for (std::size_t ip = 0; ip < header.Nbands; ip++) {
        // Process the line giving H_tb[r][ip, i].
        std::getline(fp, line);
        std::istringstream iss(line);

        // Get integer data giving r, ip, i.
        int ra, rb, rc;
        std::size_t ip_from_line, i_from_line;
        if (!(iss >> ra >> rb >> rc >> ip_from_line >> i_from_line)) {
          throw std::runtime_error("error parsing hr line");
        }
        // In file, ip and i are indexed from 1; here we index from 0.
        ip_from_line -= 1;
        i_from_line -= 1;

        if (ip_from_line != ip) {
          throw std::runtime_error("unexpected ip");
        }
        if (i_from_line != i) {
          throw std::runtime_error("unexpected i");
        }

        auto this_r = internal::process_LatVec<k_dim>(ra, rb, rc);
        if (ip == 0 and i == 0) {
          r = this_r;
        } else if (this_r != r) {
          throw std::runtime_error("unexpected variation in r");
        }

        // Get matrix element.
        double val_re, val_im;
        if (!(iss >> val_re >> val_im)) {
          throw std::runtime_error("error parsing hr line");
        }

        unsigned int degen = header.degen.at(r_index);
        std::complex<double> hr_entry(val_re / degen, val_im / degen);
        Hr(ip, i) = hr_entry;
      }
    }

    H_tb[r] = Hr;
  }

  return H_tb;
}

template <std::size_t k_dim>
TBHamiltonian<k_dim> extract_Wannier90_Hamiltonian(std::string hr_path, kComps<k_dim> Nk,
    DimMatrix<k_dim> D) {
  auto header = extract_hr_header(hr_path);
  auto H_tb = extract_hr_model<k_dim>(hr_path, header);

  return TBHamiltonian<k_dim>(H_tb, Nk, header.Nbands, D);
}

} // namespace anomtrans

#endif // ANOMTRANS_WANNIER90_HAMILTONIAN_H
