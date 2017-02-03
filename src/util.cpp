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

} // namespace anomtrans
