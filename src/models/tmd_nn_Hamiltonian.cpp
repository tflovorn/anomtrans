#include "models/tmd_nn_Hamiltonian.h"

namespace anomtrans {

namespace internal {

Eigen::MatrixXcd expand_soc(Eigen::MatrixXcd H, double lambda) {
  Eigen::MatrixXcd H_soc = Eigen::MatrixXcd::Zero(6, 6);

  Eigen::MatrixXcd Lz(3, 3);
  std::complex<double> i2(0.0, 2.0);
  Lz << 0.0, 0.0, 0.0,
        0.0, 0.0, i2,
        0.0, -i2, 0.0;

  H_soc.block<3, 3>(0, 0) = H + (lambda / 2.0) * Lz;
  H_soc.block<3, 3>(3, 3) = H - (lambda / 2.0) * Lz;

  return H_soc;
}

} // namespace internal

TMD_NN_Params get_TMD_NN_Params_WSe2() {
  return TMD_NN_Params {3.325, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338, 0.057, 0.228};
}

TBHamiltonian<2> make_TMD_NN_Hamiltonian(TMD_NN_Params p, kComps<2> Nk) {
  std::size_t Nbands = 6;

  // Tight-binding Hamiltonian matrix elements for R = 0 and R = R_1 .. R_6.
  Eigen::MatrixXcd H0(3, 3);
  Eigen::MatrixXcd HR1(3, 3), HR2(3, 3), HR3(3, 3), HR4(3, 3), HR5(3, 3), HR6(3, 3);

  H0 << p.epsilon1, 0.0, 0.0,
        0.0, p.epsilon2, 0.0,
        0.0, 0.0, p.epsilon2;

  HR1 << p.t0, p.t1, p.t2,
         -p.t1, p.t11, p.t12,
         p.t2, -p.t12, p.t22;

  double s32 = std::sqrt(3.0) / 2.0;
  double t11_22_13 = (p.t11 + 3.0 * p.t22) / 4.0;
  double t11_22_31 = (3.0 * p.t11 + p.t22) / 4.0;
  double t11_22_diff = std::sqrt(3.0) * (p.t11 - p.t22) / 4.0;

  HR2 << p.t0, -s32 * p.t2 + p.t1 / 2.0, -p.t2 / 2.0 - s32 * p.t1,
         -s32 * p.t2 - p.t1 / 2.0, t11_22_13, -t11_22_diff - p.t12,
         -p.t2 / 2.0 + s32 * p.t1, -t11_22_diff + p.t12, t11_22_31;

  HR3 << p.t0, s32 * p.t2 - p.t1 / 2.0, -p.t2 / 2.0 - s32 * p.t1,
         -s32 * p.t2 - p.t1 / 2.0, t11_22_13, t11_22_diff + p.t12,
         -p.t2 / 2.0 + s32 * p.t1, -t11_22_diff + p.t12, t11_22_31;

  HR4 << p.t0, -p.t1, p.t2,
         -p.t1, p.t11, -p.t12,
         p.t2, -p.t12, p.t22;

  HR5 << p.t0, -s32 * p.t2 - p.t1 / 2.0, -p.t2 / 2.0 + s32 * p.t1,
         -s32 * p.t2 + p.t1 / 2.0, t11_22_13, -t11_22_diff + p.t12,
         -p.t2 / 2.0 - s32 * p.t1, -t11_22_diff - p.t12, t11_22_31;

  HR6 << p.t0, s32 * p.t2 + p.t1 / 2.0, -p.t2 / 2.0 + s32 * p.t1,
         s32 * p.t2 - p.t1 / 2.0, t11_22_13, t11_22_diff - p.t12,
         -p.t2 / 2.0 - s32 * p.t1, t11_22_diff + p.t12, t11_22_31;

  std::map<LatVec<2>, Eigen::MatrixXcd> Hrs;

  Hrs[{{0, 0}}] = internal::expand_soc(H0, p.lambda);
  Hrs[{{1, 1}}] = internal::expand_soc(HR1, p.lambda);
  Hrs[{{1, 0}}] = internal::expand_soc(HR2, p.lambda);
  Hrs[{{0, -1}}] = internal::expand_soc(HR3, p.lambda);
  Hrs[{{-1, -1}}] = internal::expand_soc(HR4, p.lambda);
  Hrs[{{-1, 0}}] = internal::expand_soc(HR5, p.lambda);
  Hrs[{{0, 1}}] = internal::expand_soc(HR6, p.lambda);

  DimMatrix<2> D = {{{p.a / 2.0, p.a / 2.0},
                     {-std::sqrt(3.0) * p.a / 2.0, std::sqrt(3.0) * p.a / 2.0}}};

  return TBHamiltonian<2>(Hrs, Nk, Nbands, D);
}

} // namespace anomtrans
