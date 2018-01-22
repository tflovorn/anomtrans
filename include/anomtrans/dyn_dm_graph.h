#ifndef ANOMTRANS_DYN_DM_GRAPH_H
#define ANOMTRANS_DYN_DM_GRAPH_H

#include <cassert>
#include <string>
#include <array>
#include <map>
#include <memory>
#include <utility>
#include <boost/optional.hpp>
#include <petscksp.h>
#include "util/mat.h"
#include "grid_basis.h"
#include "observables/rho0.h"
#include "observables/energy.h"
#include "driving.h"
#include "disorder/collision_od.h"
#include "dm_graph.h"

namespace anomtrans {

/** @brief Kind of density matrix contribution: n = diagonal (n), S = off-diagonal.
 */
enum struct DynDMKind { n, S };

/** @brief Identifier for the operation used to derive a child node from a parent node.
 *  @note omega_inv_DE and Kdd_inv_DE generate diagonal (DynDMKind = n) nodes.
 *        P_inv_DE and P_inv_Kod generate off-diagonal (DynDMKind = S) nodes.
 */
enum struct DynDMDerivedBy { omega_inv_DE_up, P_inv_DE_up, Kdd_inv_DE_up, P_inv_Kod_up,
    omega_inv_DE_down, P_inv_DE_down, Kdd_inv_DE_down, P_inv_Kod_down };

/** @brief Graph node for response to a dynamic electric field (high frequency
 *         regime, omega >> 1/relaxation time).
 */
using DynDMGraphNode = DMGraphNode<DynDMKind, DynDMDerivedBy>;

boost::optional<int> get_impurity_order(boost::optional<std::shared_ptr<DynDMGraphNode>> A,
    boost::optional<std::shared_ptr<DynDMGraphNode>> B);

/** @brief Does the electric field vary as cos(\omega t), or sin(\omega t)?
 */
enum struct DynVariation { cos, sin };

/** @brief Construct the child node <rho_{n, s+1}> from the parents
 *         `lower_parent` = <rho^{(i)}_{n - 1, s}> and
 *         `upper_parent` = <rho^{(i)}_{n + 1, s}>, for n != 0.
 *         The parents are assumed to be at the same level in the expansion in powers of disorder.
 *         At least one parent must be present, but both are not required.
 */
template <std::size_t k_dim, typename Hamiltonian>
void add_dynamic_electric_n_nonzero(boost::optional<std::shared_ptr<DynDMGraphNode>> lower_parent,
    boost::optional<std::shared_ptr<DynDMGraphNode>> upper_parent, int n, double omega,
    const kmBasis<k_dim> &kmb, Mat Ehat_dot_grad_k, Mat Ehat_dot_R,
    const Hamiltonian &H, double berry_broadening, DynVariation Evar) {
  if (n == 0) {
    throw std::invalid_argument("n must be nonzero");
  }
  auto impurity_order = get_impurity_order(lower_parent, upper_parent);
  if (not impurity_order) {
    throw std::invalid_argument("lower_parent and upper_parent must have equal impurity order");
  }

  OwnedMat Dtilde = construct_driving_sum(lower_parent, upper_parent, kmb, Ehat_dot_grad_k, Ehat_dot_R, Evar);

  // Construct diagonal part of result.
  // TODO - construct n_Mat directly from Dtilde. Don't need intermediate vector Dtilde_diag.
  Vec Dtilde_diag;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm,
      &Dtilde_diag);CHKERRXX(ierr);
  ierr = MatGetDiagonal(Dtilde.M, Dtilde_diag);CHKERRXX(ierr);

  OwnedMat n_Mat = make_diag_Mat(Dtilde_diag);
  ierr = MatScale(n_Mat.M, 1.0/std::complex<double>(0.0, omega * n));CHKERRXX(ierr);

  ierr = VecDestroy(&Dtilde_diag);CHKERRXX(ierr);

  auto n_node_kind = DynDMKind::n;
  int n_impurity_order = *impurity_order;
  std::string n_name = ""; // TODO
  DynDMGraphNode::ParentsMap n_parents;
  if (lower_parent) {
    n_parents[DynDMDerivedBy::omega_inv_DE_up] = std::weak_ptr<DynDMGraphNode>(*lower_parent);
  }
  if (upper_parent) {
    n_parents[DynDMDerivedBy::omega_inv_DE_down] = std::weak_ptr<DynDMGraphNode>(*upper_parent);
  }

  auto n_node = std::make_shared<DynDMGraphNode>(std::move(n_Mat), n_node_kind, n_impurity_order,
      n_name, n_parents);

  if (lower_parent) {
    (*lower_parent)->children[DynDMDerivedBy::omega_inv_DE_up] = n_node;
  }
  if (upper_parent) {
    (*upper_parent)->children[DynDMDerivedBy::omega_inv_DE_down] = n_node;
  }

  // Construct off-diagonal part of result.
  set_Mat_diagonal(Dtilde.M, 0.0);
  OwnedMat S = apply_precession_term_dynamic(kmb, H, Dtilde.M, berry_broadening, n, omega);

  auto S_node_kind = DynDMKind::S;
  int S_impurity_order = *impurity_order;
  std::string S_name = ""; // TODO
  DynDMGraphNode::ParentsMap S_parents;
  if (lower_parent) {
    S_parents[DynDMDerivedBy::P_inv_DE_up] = std::weak_ptr<DynDMGraphNode>(*lower_parent);
  }
  if (upper_parent) {
    S_parents[DynDMDerivedBy::P_inv_DE_down] = std::weak_ptr<DynDMGraphNode>(*upper_parent);
  }

  auto S_node = std::make_shared<DynDMGraphNode>(std::move(S), S_node_kind, S_impurity_order,
      S_name, S_parents);

  if (lower_parent) {
    (*lower_parent)->children[DynDMDerivedBy::P_inv_DE_up] = S_node;
  }
  if (upper_parent) {
    (*upper_parent)->children[DynDMDerivedBy::P_inv_DE_down] = S_node;
  }
}

/** @brief Construct the child node <rho_{n, s+1}> from the parents
 *         `lower_parent` = <rho^{(i)}_{n - 1, s}> and
 *         `upper_parent` = <rho^{(i)}_{n + 1, s}>, for n = 0.
 *         The parents are assumed to be at the same level in the expansion in powers of disorder.
 *         At least one parent must be present, but both are not required.
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU_OD>
void add_dynamic_electric_n_zero(boost::optional<std::shared_ptr<DynDMGraphNode>> lower_parent,
    boost::optional<std::shared_ptr<DynDMGraphNode>> upper_parent, double omega,
    const kmBasis<k_dim> &kmb, Mat Ehat_dot_grad_k, Mat Ehat_dot_R, KSP Kdd_ksp,
    const Hamiltonian &H, const double sigma, const UU_OD &disorder_term_od,
    double berry_broadening, DynVariation Evar) {
  auto impurity_order = get_impurity_order(lower_parent, upper_parent);
  if (not impurity_order) {
    throw std::invalid_argument("lower_parent and upper_parent must have equal impurity order");
  }

  OwnedMat Dtilde = construct_driving_sum(lower_parent, upper_parent, kmb, Ehat_dot_grad_k,
      Ehat_dot_R, Evar);

  auto child_Mats = internal::get_response_electric(Dtilde.M, kmb, Kdd_ksp,
      H, sigma, disorder_term_od, berry_broadening);

  auto n_node_kind = DynDMKind::n;
  int n_impurity_order = *impurity_order - 1;
  std::string n_name = ""; // TODO
  DynDMGraphNode::ParentsMap n_parents;
  if (lower_parent) {
    n_parents[DynDMDerivedBy::Kdd_inv_DE_up] = std::weak_ptr<DynDMGraphNode>(*lower_parent);
  }
  if (upper_parent) {
    n_parents[DynDMDerivedBy::Kdd_inv_DE_down] = std::weak_ptr<DynDMGraphNode>(*upper_parent);
  }
  auto n_node = std::make_shared<DynDMGraphNode>(std::move(child_Mats[StaticDMDerivedBy::Kdd_inv_DE]),
      n_node_kind, n_impurity_order, n_name, n_parents);

  if (lower_parent) {
    (*lower_parent)->children[DynDMDerivedBy::Kdd_inv_DE_up] = n_node;
  }
  if (upper_parent) {
    (*upper_parent)->children[DynDMDerivedBy::Kdd_inv_DE_down] = n_node;
  }

  auto S_intrinsic_node_kind = DynDMKind::S;
  int S_intrinsic_impurity_order = *impurity_order;
  std::string S_intrinsic_name = ""; // TODO intrinsic vs extrinsic in name
  DynDMGraphNode::ParentsMap S_intrinsic_parents;
  if (lower_parent) {
    S_intrinsic_parents[DynDMDerivedBy::P_inv_DE_up] = std::weak_ptr<DynDMGraphNode>(*lower_parent);
  }
  if (upper_parent) {
    S_intrinsic_parents[DynDMDerivedBy::P_inv_DE_down] = std::weak_ptr<DynDMGraphNode>(*upper_parent);
  }
  auto S_intrinsic_node = std::make_shared<DynDMGraphNode>(std::move(child_Mats[StaticDMDerivedBy::P_inv_DE]),
      S_intrinsic_node_kind, S_intrinsic_impurity_order, S_intrinsic_name, S_intrinsic_parents);

  if (lower_parent) {
    (*lower_parent)->children[DynDMDerivedBy::P_inv_DE_up] = S_intrinsic_node;
  }
  if (upper_parent) {
    (*upper_parent)->children[DynDMDerivedBy::P_inv_DE_down] = S_intrinsic_node;
  }

  auto S_extrinsic_node_kind = DynDMKind::S;
  int S_extrinsic_impurity_order = *impurity_order;
  std::string S_extrinsic_name = ""; // TODO
  DynDMGraphNode::ParentsMap S_extrinsic_parents;
  if (lower_parent) {
    S_extrinsic_parents[DynDMDerivedBy::P_inv_Kod_up] = std::weak_ptr<DynDMGraphNode>(*lower_parent);
  }
  if (upper_parent) {
    S_extrinsic_parents[DynDMDerivedBy::P_inv_Kod_down] = std::weak_ptr<DynDMGraphNode>(*upper_parent);
  }
  auto S_extrinsic_node = std::make_shared<DynDMGraphNode>(std::move(child_Mats[StaticDMDerivedBy::P_inv_Kod]),
      S_extrinsic_node_kind, S_extrinsic_impurity_order, S_extrinsic_name, S_extrinsic_parents);

  if (lower_parent) {
    (*lower_parent)->children[DynDMDerivedBy::P_inv_Kod_up] = S_extrinsic_node;
  }
  if (upper_parent) {
    (*upper_parent)->children[DynDMDerivedBy::P_inv_Kod_down] = S_extrinsic_node;
  }
}

namespace internal {

std::complex<double> get_driving_sum_scale(DynVariation Evar);

std::complex<double> get_driving_sum_upper_factor(DynVariation Evar);

} // namespace internal

/** @brief Construct the driving term
 *         \tilde{D}^{(cos)}_{n,s} = (1/2) [D_{E_0}(<rho_{n-1, s}>) + D_{E_0}(<rho_{n+1, s}>)]
 *         or
 *         \tilde{D}^{(sin)}_{n,s} = (1/2i) [D_{E_0}(<rho_{n-1, s}>) - D_{E_0}(<rho_{n+1, s}>)]
 *         from the given parent nodes
 *         `lower_parent` = <rho^{(i)}_{n - 1, s}> and
 *         `upper_parent` = <rho^{(i)}_{n + 1, s}>.
 *         The parents are assumed to be at the same level in the expansion in powers of disorder.
 *         At least one parent must be present, but both are not required.
 */
template <std::size_t k_dim>
OwnedMat construct_driving_sum(boost::optional<std::shared_ptr<DynDMGraphNode>> lower_parent,
    boost::optional<std::shared_ptr<DynDMGraphNode>> upper_parent, const kmBasis<k_dim> &kmb,
    Mat Ehat_dot_grad_k, Mat Ehat_dot_R, DynVariation Evar) {
  auto driving_sum_scale = internal::get_driving_sum_scale(Evar);
  auto driving_sum_upper_factor = internal::get_driving_sum_upper_factor(Evar);

  if (lower_parent and upper_parent) {
    OwnedMat D_E_lower = apply_driving_electric(kmb, Ehat_dot_grad_k, Ehat_dot_R, ((*lower_parent)->rho).M);
    OwnedMat D_E_upper = apply_driving_electric(kmb, Ehat_dot_grad_k, Ehat_dot_R, ((*upper_parent)->rho).M);

    PetscErrorCode ierr = MatAXPY(D_E_lower.M, driving_sum_upper_factor, D_E_upper.M, DIFFERENT_NONZERO_PATTERN);CHKERRXX(ierr);
    ierr = MatScale(D_E_lower.M, driving_sum_scale);CHKERRXX(ierr);

    return D_E_lower;
  } else if (lower_parent) {
    OwnedMat D_E_val = apply_driving_electric(kmb, Ehat_dot_grad_k, Ehat_dot_R, ((*lower_parent)->rho).M);
    PetscErrorCode ierr = MatScale(D_E_val.M, driving_sum_scale);CHKERRXX(ierr);
    return D_E_val;
  } else if (upper_parent) {
    OwnedMat D_E_val = apply_driving_electric(kmb, Ehat_dot_grad_k, Ehat_dot_R, ((*upper_parent)->rho).M);
    PetscErrorCode ierr = MatScale(D_E_val.M, driving_sum_scale * driving_sum_upper_factor);CHKERRXX(ierr);
    return D_E_val;
  } else {
    throw std::invalid_argument("at least one of lower_parent and upper_parent must be specified");
  }
}

} // namespace anomtrans

#endif // ANOMTRANS_DYN_DM_GRAPH_H
