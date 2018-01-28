#ifndef ANOMTRANS_DM_GRAPH_H
#define ANOMTRANS_DM_GRAPH_H

#include <stdexcept>
#include <string>
#include <array>
#include <map>
#include <memory>
#include <utility>
#include <petscksp.h>
#include "util/mat.h"
#include "grid_basis.h"
#include "observables/rho0.h"
#include "driving.h"
#include "disorder/collision_od.h"

namespace anomtrans {

template <typename NodeMatKind, typename NodeDerivedBy>
class DMGraphNode {
public:
  using MatKindType = NodeMatKind;
  using DerivedByType = NodeDerivedBy;

  using ParentsMap = std::map<NodeDerivedBy, std::weak_ptr<DMGraphNode<NodeMatKind, NodeDerivedBy>>>;
  using ChildrenMap = std::map<NodeDerivedBy, std::shared_ptr<DMGraphNode<NodeMatKind, NodeDerivedBy>>>;

  /** @brief Density matrix given by this node.
   */
  OwnedMat rho;

  const NodeMatKind node_kind;

  /** @brief Order of this node in the expansion in powers of impurity density.
   */
  const int impurity_order;

  /** @brief Identifier for this node, constructed from its kind, impurity order,
   *         and derivation sequence.
   */
  const std::string name;

  const ParentsMap parents;
  ChildrenMap children;

  /** @note When this node is constructed, it takes ownership of the matrix rho.
   *  @note The map of children is default-initialized, i.e. it starts empty.
   *  @todo Could initialize node_kind, impurity_order, name based on parents
   *        instead of passing these in here.
   */
  DMGraphNode(OwnedMat&& _rho, NodeMatKind _node_kind, int _impurity_order, std::string _name,
      ParentsMap _parents)
      : rho(std::move(_rho)), node_kind(_node_kind), impurity_order(_impurity_order),
        name(_name), parents(_parents) {}

  /** @brief Returns an ordered map containing this node and all its descendants,
   *         without duplicates. The keys of the map are given by the names of the nodes.
   *         The nodes are ordered by increasing impurity order.
   */
  //std::map<std::string, std::shared_ptr<DMGraphNode<NodeMatKind, NodeDerivedBy>>> unique_descendants();
};

/** @brief Kind of density matrix contribution: n = diagonal (n), S = off-diagonal,
 *         xi = diagonal via P^{-1} D_B ~ B dot Omega.
 */
enum struct StaticDMKind { n, S, xi };

/** @brief Identifier for the operation used to derive a child node from a parent node.
 */
enum struct StaticDMDerivedBy { P_inv_DB, B_dot_Omega, Kdd_inv_DE, P_inv_DE, P_inv_Kod, Kdd_inv_DB };

/** @brief Graph node for linear response to a static electric field and nonlinear
 *         response to a weak static magnetic field.
 */
using StaticDMGraphNode = DMGraphNode<StaticDMKind, StaticDMDerivedBy>;

/** @brief Construct the node containing the equilibrium density matrix
 *         (which has no parent nodes).
 *  @todo Would prefer to return pair<DMGraphNode, Vec> including rho0_km,
 *        since this Vec is required to construct the nullspace of Kdd.
 *        Tricky using tie due to intermediate default construction.
 *        Works better with structured bindings?
 */
template <typename DMNodeType>
std::shared_ptr<DMNodeType> make_eq_node(Vec Ekm, double beta, double mu) {
    auto rho0_km = make_rho0(Ekm, beta, mu);
    OwnedMat rho0_km_Mat = make_diag_Mat(rho0_km.v);

    auto node_kind = DMNodeType::MatKindType::n;
    int impurity_order = 0;
    std::string name = "\\rho_0";
    typename DMNodeType::ParentsMap parents;

    auto rho0_node = std::make_shared<DMNodeType>(std::move(rho0_km_Mat), node_kind,
        impurity_order, name, parents);

    return rho0_node;
}

namespace internal {

template <std::size_t k_dim, typename Hamiltonian, typename UU_OD>
std::map<StaticDMDerivedBy, OwnedMat> get_response_electric(Mat D_E_rho,
    const kmBasis<k_dim> &kmb, KSP Kdd_ksp,
    const Hamiltonian &H, const double sigma, const UU_OD &disorder_term_od,
    double berry_broadening) {
  // Construct <n_E^(-1)> = Kdd^{-1} D_E (<rho>).
  auto D_E_rho_diag = make_Vec(kmb.end_ikm);
  PetscErrorCode ierr = MatGetDiagonal(D_E_rho, D_E_rho_diag.v);CHKERRXX(ierr);

  auto n_E = make_Vec_with_structure(D_E_rho_diag.v);
  ierr = KSPSolve(Kdd_ksp, D_E_rho_diag.v, n_E.v);CHKERRXX(ierr);

  OwnedMat n_E_Mat = make_diag_Mat(n_E.v);

  // Construct <S_E>_k^{mm'} = [P^{-1} [D_E(<rho>) - K^{od}(<n_E>)]]_k^{mm'}
  //   = -i\hbar [D_E(<rho>) - K^{od}(<n_E>)]]_k^{mm'} / (E_{km} - E_{km'})
  //   \approx -i\hbar [D_E(<rho>) - K^{od}(<n_E>)]]_k^{mm'}
  //               * (E_{km} - E_{km'}) / ((E_{km} - E_{km'})^2 + \eta^2)
  // Here \eta is the broadening applied to treat degeneracies, chosen to be the same
  // as the broadening used in the calculation of the Berry connection.
  // Keep the intrinsic P^{-1} D_E(<rho_0>) and extrinsic P^{-1} K^{od}(<n_E^(-1)>)
  // parts separate.
  // Intrinsic part of S:
  set_Mat_diagonal(D_E_rho, 0.0);
  OwnedMat S_E_intrinsic = apply_precession_term(kmb, H, D_E_rho, berry_broadening);

  // Extrinsic part of S:
  auto n_E_all = scatter_to_all(n_E.v);
  auto n_E_all_std = std::get<1>(get_local_contents(n_E_all.v));
  OwnedMat K_od_n_E = apply_collision_od(kmb, H, sigma, disorder_term_od, n_E_all_std);
  ierr = MatScale(K_od_n_E.M, -1.0);CHKERRXX(ierr);

  OwnedMat S_E_extrinsic = apply_precession_term(kmb, H, K_od_n_E.M, berry_broadening);

  std::map<StaticDMDerivedBy, OwnedMat> result;
  result.insert(std::make_pair(StaticDMDerivedBy::Kdd_inv_DE, std::move(n_E_Mat)));
  result.insert(std::make_pair(StaticDMDerivedBy::P_inv_DE, std::move(S_E_intrinsic)));
  result.insert(std::make_pair(StaticDMDerivedBy::P_inv_Kod, std::move(S_E_extrinsic)));

  return result;
}

} // namespace internal

/** @brief Given a node containing the equilibrium density matrix <rho_0> or the diagonal
 *         part of the magnetic field linear response <xi_B>,
 *         add children to it corresponding to linear response to an applied electric
 *         field and return a list of those children.
 *  @param disorder_term_od A function with signature `complex<double> f(ikm1, ikm2, ikm3)`
 *                          giving the disorder-averaged term U_{ikm1, ikm2} U_{ikm2, ikm3}.
 *                          Must have k1 = k3 for valid result.
 *  @precondition Kdd_ksp should have its nullspace set appropriately before this function is
 *                called (the nullspace of Kdd is the <rho_0> vector).
 *  @todo Verify that parent_node is appropriate. <rho_0> or <xi_B> are allowed.
 *        Should <S_B> be allowed?
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU_OD>
void add_linear_response_electric(std::shared_ptr<StaticDMGraphNode> parent_node,
    const kmBasis<k_dim> &kmb, Mat Ehat_dot_grad_k, Mat Ehat_dot_R, KSP Kdd_ksp,
    const Hamiltonian &H, const double sigma, const UU_OD &disorder_term_od,
    double berry_broadening) {
  OwnedMat D_E_rho = apply_driving_electric(kmb, Ehat_dot_grad_k, Ehat_dot_R, parent_node->rho.M);

  auto child_Mats = internal::get_response_electric(D_E_rho.M, kmb, Kdd_ksp,
      H, sigma, disorder_term_od, berry_broadening);

  auto n_E_node_kind = StaticDMKind::n;
  int n_E_impurity_order = parent_node->impurity_order - 1;
  std::string n_E_name = ""; // TODO
  StaticDMGraphNode::ParentsMap n_E_parents {
    {StaticDMDerivedBy::Kdd_inv_DE, std::weak_ptr<StaticDMGraphNode>(parent_node)}
  };
  auto n_E_node = std::make_shared<StaticDMGraphNode>(std::move(child_Mats[StaticDMDerivedBy::Kdd_inv_DE]),
      n_E_node_kind, n_E_impurity_order, n_E_name, n_E_parents);

  parent_node->children[StaticDMDerivedBy::Kdd_inv_DE] = n_E_node;

  auto S_E_intrinsic_node_kind = StaticDMKind::S;
  int S_E_intrinsic_impurity_order = parent_node->impurity_order;
  std::string S_E_intrinsic_name = ""; // TODO
  StaticDMGraphNode::ParentsMap S_E_intrinsic_parents {
    {StaticDMDerivedBy::P_inv_DE, std::weak_ptr<StaticDMGraphNode>(parent_node)},
  };
  auto S_E_intrinsic_node = std::make_shared<StaticDMGraphNode>(std::move(child_Mats[StaticDMDerivedBy::P_inv_DE]),
      S_E_intrinsic_node_kind, S_E_intrinsic_impurity_order, S_E_intrinsic_name,
      S_E_intrinsic_parents);

  parent_node->children[StaticDMDerivedBy::P_inv_DE] = S_E_intrinsic_node;

  auto S_E_extrinsic_node_kind = StaticDMKind::S;
  int S_E_extrinsic_impurity_order = parent_node->impurity_order;
  std::string S_E_extrinsic_name = ""; // TODO
  StaticDMGraphNode::ParentsMap S_E_extrinsic_parents {
    {StaticDMDerivedBy::P_inv_Kod, std::weak_ptr<StaticDMGraphNode>(n_E_node)}
  };
  auto S_E_extrinsic_node = std::make_shared<StaticDMGraphNode>(std::move(child_Mats[StaticDMDerivedBy::P_inv_Kod]),
      S_E_extrinsic_node_kind, S_E_extrinsic_impurity_order, S_E_extrinsic_name,
      S_E_extrinsic_parents);

  n_E_node->children[StaticDMDerivedBy::P_inv_Kod] = S_E_extrinsic_node;
}

/** @brief Given a node containing the equilibrium density matrix,
 *         add children to it corresponding to linear response to an applied magnetic
 *         field and return a list of those children.
 */
template <std::size_t k_dim, typename Hamiltonian>
void add_linear_response_magnetic(std::shared_ptr<StaticDMGraphNode> eq_node,
    const kmBasis<k_dim> &kmb, std::array<OwnedMat, k_dim>& DH0_cross_Bhat,
    std::array<OwnedMat, k_dim>& d_dk_Cart, std::array<OwnedMat, k_dim>& R,
    Vec Bhat_dot_Omega, const Hamiltonian &H, double berry_broadening) {
  // eq_node must be the equilibrium density matrix, which has no parents.
  if (eq_node->parents.size() != 0) {
    throw std::invalid_argument("must apply add_linear_response_magnetic to equilibrium node");
  }

  // Construct off-diagonal response, <S_B^{(0)}> = [P^{-1} D_B(<rho_0>)]_{m, m' \neq m}.
  OwnedMat D_B_rho = apply_driving_magnetic(kmb, DH0_cross_Bhat, d_dk_Cart, R, eq_node->rho.M);

  set_Mat_diagonal(D_B_rho.M, 0.0);
  OwnedMat child_S_B_intrinsic = apply_precession_term(kmb, H, D_B_rho.M, berry_broadening);

  auto child_S_B_intrinsic_node_kind = StaticDMKind::S;
  int child_S_B_intrinsic_impurity_order = 0;
  std::string child_S_B_intrinsic_name = "S_B^{(0)}";
  StaticDMGraphNode::ParentsMap child_S_B_intrinsic_parents {
    {StaticDMDerivedBy::P_inv_DB, std::weak_ptr<StaticDMGraphNode>(eq_node)},
  };
  auto child_S_B_intrinsic_node = std::make_shared<StaticDMGraphNode>(std::move(child_S_B_intrinsic),
      child_S_B_intrinsic_node_kind, child_S_B_intrinsic_impurity_order,
      child_S_B_intrinsic_name, child_S_B_intrinsic_parents);

  eq_node->children[StaticDMDerivedBy::P_inv_DB] = child_S_B_intrinsic_node;

  // Construct diagonal response, <\xi_B^{(0)}> = [P^{-1} D_B(<rho_0>)]_{mm}
  //  = (e / \hbar) n_{km} * (B dot Omega_{km}).
  auto eq_diag = make_Vec(kmb.end_ikm);
  PetscErrorCode ierr = MatGetDiagonal(eq_node->rho.M, eq_diag.v);CHKERRXX(ierr);

  auto xi = make_Vec_with_structure(eq_diag.v);
  ierr = VecPointwiseMult(xi.v, eq_diag.v, Bhat_dot_Omega);CHKERRXX(ierr);

  OwnedMat child_xi_Mat = make_diag_Mat(xi.v);
  auto child_xi_node_kind = StaticDMKind::xi;
  int child_xi_impurity_order = 0;
  std::string child_xi_name = "\\xi_B^{(0)}";
  StaticDMGraphNode::ParentsMap child_xi_parents {
    {StaticDMDerivedBy::B_dot_Omega, std::weak_ptr<StaticDMGraphNode>(eq_node)}
  };
  auto child_xi_node = std::make_shared<StaticDMGraphNode>(std::move(child_xi_Mat),
      child_xi_node_kind, child_xi_impurity_order, child_xi_name, child_xi_parents);

  eq_node->children[StaticDMDerivedBy::B_dot_Omega] = child_xi_node;
}

/** @brief Add children to `node` corresponding to the next order of the expansion in powers
 *         of magnetic field and return a list of those children.
 *  @precondition `node` should already include the electric field linear response,
 *                i.e. it should be <n_E^{(-1)}>, <S_E^{(0)}>, or one of their descendants.
 *  @precondition Kdd_ksp should have its nullspace set appropriately before this function is
 *                called (the nullspace of Kdd is the <rho_0> vector).
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU_OD>
void add_next_order_magnetic(std::shared_ptr<StaticDMGraphNode> parent_node,
    const kmBasis<k_dim> &kmb, std::array<OwnedMat, k_dim>& DH0_cross_Bhat,
    std::array<OwnedMat, k_dim>& d_dk_Cart, std::array<OwnedMat, k_dim>& R, KSP Kdd_ksp,
    Vec Bhat_dot_Omega, const Hamiltonian &H, const double sigma, const UU_OD &disorder_term_od,
    double berry_broadening) {
  // TODO check that parent_node has the appropriate structure
  // (it should be <rho>_{E, B^{N-1}}).

  // Construct n child: Kdd^{-1} D_B (<rho>).
  OwnedMat D_B_rho = apply_driving_magnetic(kmb, DH0_cross_Bhat, d_dk_Cart, R, parent_node->rho.M);

  auto D_B_rho_diag = make_Vec(kmb.end_ikm);
  PetscErrorCode ierr = MatGetDiagonal(D_B_rho.M, D_B_rho_diag.v);CHKERRXX(ierr);

  auto child_n_B = make_Vec_with_structure(D_B_rho_diag.v);
  ierr = KSPSolve(Kdd_ksp, D_B_rho_diag.v, child_n_B.v);CHKERRXX(ierr);

  OwnedMat child_n_B_Mat = make_diag_Mat(child_n_B.v);
  auto child_n_B_node_kind = StaticDMKind::n;
  int child_n_B_impurity_order = parent_node->impurity_order - 1;
  std::string child_n_B_name = ""; // TODO
  StaticDMGraphNode::ParentsMap child_n_B_parents {
    {StaticDMDerivedBy::Kdd_inv_DB, std::weak_ptr<StaticDMGraphNode>(parent_node)}
  };
  auto child_n_B_node = std::make_shared<StaticDMGraphNode>(std::move(child_n_B_Mat),
      child_n_B_node_kind, child_n_B_impurity_order, child_n_B_name, child_n_B_parents);

  parent_node->children[StaticDMDerivedBy::Kdd_inv_DB] = child_n_B_node;

  // Construct <S_EB^(N)>_k^{mm'} = [P^{-1} [D_B(<rho_EB^{N-1}>) - K^{od}(<n_EB^N>)]]_k^{mm'}
  //   = -i\hbar [D_E(<rho_EB^{N-1}>) - K^{od}(<n_EB^N>)]]_k^{mm'} / (E_{km} - E_{km'})
  //   \approx -i\hbar [D_E(<rho_EB^{N-1}>) - K^{od}(<n_EB^N>)]]_k^{mm'}
  //               * (E_{km} - E_{km'}) / ((E_{km} - E_{km'})^2 + \eta^2)
  // Here \eta is the broadening applied to treat degeneracies, chosen to be the same
  // as the broadening used in the calculation of the Berry connection.
  // Keep the intrinsic P^{-1} D_B(<rho_EB^(N-1)> and extrinsic P^{-1} K^{od}(<n_EB^(N)>)
  // parts separate.
  // Intrinsic part of S:
  set_Mat_diagonal(D_B_rho.M, 0.0);
  OwnedMat child_S_B_intrinsic = apply_precession_term(kmb, H, D_B_rho.M, berry_broadening);

  auto child_S_B_intrinsic_node_kind = StaticDMKind::S;
  int child_S_B_intrinsic_impurity_order = parent_node->impurity_order;
  std::string child_S_B_intrinsic_name = ""; // TODO
  StaticDMGraphNode::ParentsMap child_S_B_intrinsic_parents {
    {StaticDMDerivedBy::P_inv_DB, std::weak_ptr<StaticDMGraphNode>(parent_node)},
  };
  auto child_S_B_intrinsic_node = std::make_shared<StaticDMGraphNode>(std::move(child_S_B_intrinsic),
      child_S_B_intrinsic_node_kind, child_S_B_intrinsic_impurity_order,
      child_S_B_intrinsic_name, child_S_B_intrinsic_parents);

  parent_node->children[StaticDMDerivedBy::P_inv_DB] = child_S_B_intrinsic_node;

  // Extrinsic part of S:
  auto child_n_B_all = scatter_to_all(child_n_B.v);
  auto child_n_B_all_std = std::get<1>(get_local_contents(child_n_B_all.v));
  OwnedMat K_od_child_n_B = apply_collision_od(kmb, H, sigma, disorder_term_od, child_n_B_all_std);
  ierr = MatScale(K_od_child_n_B.M, -1.0);CHKERRXX(ierr);

  OwnedMat child_S_B_extrinsic = apply_precession_term(kmb, H, K_od_child_n_B.M, berry_broadening);

  auto child_S_B_extrinsic_node_kind = StaticDMKind::S;
  int child_S_B_extrinsic_impurity_order = parent_node->impurity_order;
  std::string child_S_B_extrinsic_name = ""; // TODO
  StaticDMGraphNode::ParentsMap child_S_B_extrinsic_parents {
    {StaticDMDerivedBy::P_inv_Kod, std::weak_ptr<StaticDMGraphNode>(child_n_B_node)},
  };
  auto child_S_B_extrinsic_node = std::make_shared<StaticDMGraphNode>(std::move(child_S_B_extrinsic),
      child_S_B_extrinsic_node_kind, child_S_B_extrinsic_impurity_order,
      child_S_B_extrinsic_name, child_S_B_extrinsic_parents);

  child_n_B_node->children[StaticDMDerivedBy::P_inv_Kod] = child_S_B_extrinsic_node;

  if (parent_node->node_kind != StaticDMKind::S) {
    // Construct xi child.
    // xi and n have xi children, but S does not.
    auto parent_diag = make_Vec_with_structure(D_B_rho_diag.v);
    ierr = MatGetDiagonal(parent_node->rho.M, parent_diag.v);CHKERRXX(ierr);

    // xi_{km} = n_{km} * (B dot Omega_{km})
    auto xi = make_Vec_with_structure(D_B_rho_diag.v);
    ierr = VecPointwiseMult(xi.v, parent_diag.v, Bhat_dot_Omega);CHKERRXX(ierr);

    OwnedMat child_xi_Mat = make_diag_Mat(xi.v);
    auto child_xi_node_kind = StaticDMKind::xi;
    int child_xi_impurity_order = parent_node->impurity_order;
    std::string child_xi_name = ""; // TODO
    StaticDMGraphNode::ParentsMap child_xi_parents {
      {StaticDMDerivedBy::B_dot_Omega, std::weak_ptr<StaticDMGraphNode>(parent_node)}
    };
    auto child_xi_node = std::make_shared<StaticDMGraphNode>(std::move(child_xi_Mat),
        child_xi_node_kind, child_xi_impurity_order, child_xi_name, child_xi_parents);

    parent_node->children[StaticDMDerivedBy::B_dot_Omega] = child_xi_node;
  }
}

} // namespace anomtrans

#endif // ANOMTRANS_DM_GRAPH_H
