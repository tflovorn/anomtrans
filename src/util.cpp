#include "util.h"

namespace anomtrans {

std::vector<double> linspace(double start, double stop, unsigned int num) {
  std::vector<double> v;
  v.reserve(num);
  
  if (num == 0) {
    return v;
  }
  
  double x = start;
  v.push_back(x);
  
  if (num == 1) {
    return v;
  }
  
  double step = (stop - start) / (num - 1);
  for (unsigned int i = 1; i < num; i++) {
    x += step;
    v.push_back(x);
  }
    
  return v;
}

boost::optional<std::string> getenv_optional(const std::string& var) {
  // The pointer returned by std::getenv is 'internal memory' which we are
  // not allowed to modify. We do not leak memory by not deleting it.
  // TODO - sure this is correct?
  const char* val = std::getenv(var.c_str());
  if (val == nullptr) {
    return boost::none;
  } else {
    return std::string(val);
  }
}

bool check_json_equal(std::string test_path, std::string known_path) {
  // Want to load this from a file instead of comparing calculated test to
  // stored known to avoid floating-point comparison errors.
  nlohmann::json j_test;
  std::ifstream fp_t(test_path);
  if (not fp_t.good()) {
    throw std::runtime_error("could not open file in check_json_equal");
  }
  fp_t >> j_test;
  fp_t.close();

  nlohmann::json j_known;
  std::ifstream fp_k(known_path);
  if (not fp_k.good()) {
    throw std::runtime_error("could not open file in check_json_equal");
  }
  fp_k >> j_known;
  fp_k.close();

  return j_test == j_known;
}

template <>
bool check_equal_within<PetscReal>(std::vector<PetscReal> xs, std::vector<PetscReal> ys, PetscReal tol) {
  assert(tol > 0.0);
  if (xs.size() != ys.size()) {
    return false;
  }
  for (std::vector<PetscReal>::size_type i = 0; i < xs.size(); i++) {
    if (std::abs(xs.at(i) - ys.at(i)) > tol) {
      return false;
    }
  }
  return true;
}

} // namespace anomtrans
