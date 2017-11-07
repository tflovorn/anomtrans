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
#include "util/lattice.h"
#include "grid_basis.h"

namespace anomtrans {

/** @brief Tight-binding Hamiltonian.
 */
template <std::size_t k_dim>
class TBHamiltonian {
public:
  /** @brief Tight-binding Hamiltonian matrix. 
   */
  const std::map<LatVec<k_dim>, Eigen::MatrixXcd> Hrs;

  /** @brief Discretization of (k, m) basis to use for Fourier transform.
   */
  const kmBasis<k_dim> kmb;

  TBHamiltonian(std::map<LatVec<k_dim>, Eigen::MatrixXcd> _Hrs, kComps<k_dim> _Nk,
      std::size_t _Nbands) : Hrs(_Hrs), kmb(kmBasis<k_dim>(_Nk, _Nbands)) {}
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
TBHamiltonian<k_dim> extract_Wannier90_Hamiltonian(std::string hr_path, kComps<k_dim> Nk) {
  auto header = extract_hr_header(hr_path);
  auto H_tb = extract_hr_model<k_dim>(hr_path, header);

  return TBHamiltonian<k_dim>(H_tb, Nk, header.Nbands);
}

} // namespace anomtrans

#endif // ANOMTRANS_WANNIER90_HAMILTONIAN_H
