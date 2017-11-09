#include "models/wannier90_Hamiltonian.h"

namespace anomtrans {

HrHeader extract_hr_header(const std::string hr_path) {
  std::ifstream fp(hr_path);

  std::string line;
  // Skip line 0, comment line.
  std::getline(fp, line);
  // Get Nbands (line 1) and Nrs (line 2).
  std::size_t Nbands, Nrs;
  std::getline(fp, line);
  std::istringstream iss(line);
  if (!(iss >> Nbands)) {
    throw std::runtime_error("error extracting Nbands");
  }
  if (Nbands == 0) {
    throw std::runtime_error("must have at least one band");
  }

  std::getline(fp, line);
  iss = std::istringstream(line);
  if (!(iss >> Nrs)) {
    throw std::runtime_error("error extracting Nrs");
  }
  if (Nrs == 0) {
    throw std::runtime_error("must have at least one r");
  }
  // Get remaining lines in header, giving up to 15 degen values per line,
  // starting at line 3.
  std::vector<unsigned int> degen;
  for (std::size_t line_number = 3; degen.size() < Nrs; line_number++) {
    std::getline(fp, line);
    iss = std::istringstream(line);

    for (std::size_t rs_this_line = 0; rs_this_line < 15 and degen.size() < Nrs; rs_this_line++) {
      unsigned int this_degen;
      if (!(iss >> this_degen)) {
        throw std::runtime_error("error extracting degen");
      }
      degen.push_back(this_degen);
    }
  }
  if (degen.size() != Nrs) {
    // sanity check
    throw std::runtime_error("got unexpected number of rs");
  }

  std::size_t num_degen_lines = 1 + (Nrs - 1) / 15;
  std::size_t start_hr = 2 + num_degen_lines + 1;

  return HrHeader{Nbands, Nrs, degen, start_hr};
}

std::array<Eigen::MatrixXcd, 3> full_spin_matrices(const std::size_t Nbands) {
  if (Nbands % 2 != 0) {
    throw std::invalid_argument("expected Nbands divisible by 2 (both spins present)");
  }

	auto pauli = pauli_matrices();
  std::array<Eigen::MatrixXcd, 3> S;
	for (std::size_t dc = 0; dc < 3; dc++) {
		S.at(dc) = Eigen::MatrixXcd(Nbands, Nbands);
		for (std::size_t i = 0; i < Nbands; i += 2) {
			S.at(dc).block(i, i, 2, 2) = pauli.at(dc);
		}
	}

  return S;
}

namespace internal {

template <>
LatVec<1> process_LatVec(int ra, int rb, int rc) {
  return {ra};
}

template <>
LatVec<2> process_LatVec(int ra, int rb, int rc) {
  return {ra, rb};
}

template <>
LatVec<3> process_LatVec(int ra, int rb, int rc) {
  return {ra, rb, rc};
}

} // namespace internal

} // namespace anomtrans
