#ifndef ANOMTRANS_DM_GRAPH_H
#define ANOMTRANS_DM_GRAPH_H

#include <cassert>
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

  /** @brief Density matrix given by this node. This matrix is owned by this node,
   *         i.e. the node is responsible for destroying it.
   *  @todo Would like to expose only const Mat, but many PETSc functions which are logically
   *        const do not take const Mat (e.g. MatMult). Could protect this value by implementing
   *        pass-through methods for required PETSc functions.
   */
  Mat rho;

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
  DMGraphNode(Mat _rho, NodeMatKind _node_kind, int _impurity_order, std::string _name,
      ParentsMap _parents)
      : rho(_rho), node_kind(_node_kind), impurity_order(_impurity_order),
        name(_name), parents(_parents) {}

  ~DMGraphNode() {
    PetscErrorCode ierr = MatDestroy(&rho);CHKERRXX(ierr);
    // TODO other cleanup required?
  }

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
    Vec rho0_km = make_rho0(Ekm, beta, mu);
    Mat rho0_km_Mat = make_diag_Mat(rho0_km);

    auto node_kind = DMNodeType::MatKindType::n;
    int impurity_order = 0;
    std::string name = "\\rho_0";
    typename DMNodeType::ParentsMap parents;

    auto rho0_node = std::make_shared<DMNodeType>(rho0_km_Mat, node_kind, impurity_order, name, parents);

    PetscErrorCode ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);

    return rho0_node;
}

/** @brief Given a node containing the equilibrium density matrix,
 *         add children to it corresponding to linear response to an applied magnetic
 *         field and return a list of those children.
 *  @todo Add parameters necessary to implement this.
 *  @todo Implementation can check that rho0 is really the equilibrium density matrix
 *        by verifying that its list of parents is empty.
 */
//DMGraphNode::ChildrenMap add_linear_response_magnetic(DMGraphNode &eq_node);

/** @brief Given a node containing the equilibrium density matrix,
 *         add children to it corresponding to linear response to an applied electric
 *         field and return a list of those children.
 *  @param disorder_term_od A function with signature `complex<double> f(ikm1, ikm2, ikm3)`
 *                          giving the disorder-averaged term U_{ikm1, ikm2} U_{ikm2, ikm3}.
 *                          Must have k1 = k3 for valid result.
 *  @precondition Kdd_ksp should have its nullspace set appropriately before this function is
 *                called (the nullspace of Kdd is the <rho_0> vector).
 */
template <std::size_t k_dim, typename Hamiltonian, typename UU_OD>
void add_linear_response_electric(std::shared_ptr<StaticDMGraphNode> eq_node,
    const kmBasis<k_dim> &kmb, Mat Ehat_dot_grad_k, Mat Ehat_dot_R, KSP Kdd_ksp,
    const Hamiltonian &H, const double sigma, const UU_OD &disorder_term_od,
    double berry_broadening) {
  // eq_node must be the equilibrium density matrix, which has no parents.
  assert(eq_node->parents.size() == 0);

  // Construct <n_E^(-1)> = Kdd^{-1} D_E (<rho_0>).
  Mat D_E_rho0 = apply_driving_electric(kmb, Ehat_dot_grad_k, Ehat_dot_R, eq_node->rho);

  Vec D_E_rho0_diag;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm,
      &D_E_rho0_diag);CHKERRXX(ierr);
  ierr = MatGetDiagonal(D_E_rho0, D_E_rho0_diag);CHKERRXX(ierr);

  Vec n_E;
  ierr = VecDuplicate(D_E_rho0_diag, &n_E);CHKERRXX(ierr);
  ierr = KSPSolve(Kdd_ksp, D_E_rho0_diag, n_E);CHKERRXX(ierr);

  Mat n_E_Mat = make_diag_Mat(n_E);
  auto n_E_node_kind = StaticDMKind::n;
  int n_E_impurity_order = -1;
  std::string n_E_name = "n_E^{(-1)}";
  StaticDMGraphNode::ParentsMap n_E_parents {
    {StaticDMDerivedBy::Kdd_inv_DE, std::weak_ptr<StaticDMGraphNode>(eq_node)}
  };
  auto n_E_node = std::make_shared<StaticDMGraphNode>(n_E_Mat, n_E_node_kind, n_E_impurity_order,
      n_E_name, n_E_parents);

  eq_node->children[StaticDMDerivedBy::Kdd_inv_DE] = n_E_node;

  // Construct <S_E^(0)>_k^{mm'} = [P^{-1} [D_E(<rho_0>) - K^{od}(<n_E>)]]_k^{mm'}
  //   = -i\hbar [D_E(<rho_0>) - K^{od}(<n_E>)]]_k^{mm'} / (E_{km} - E_{km'})
  //   \approx -i\hbar [D_E(<rho_0>) - K^{od}(<n_E>)]]_k^{mm'}
  //               * (E_{km} - E_{km'}) / ((E_{km} - E_{km'})^2 + \eta^2)
  // Here \eta is the broadening applied to treat degeneracies, chosen to be the same
  // as the broadening used in the calculation of the Berry connection.
  // Keep the intrinsic P^{-1} D_E(<rho_0>) and extrinsic P^{-1} K^{od}(<n_E^(-1)>)
  // parts separate.
  // Intrinsic part of S:
  set_Mat_diagonal(D_E_rho0, 0.0);
  Mat S_E_intrinsic = apply_precession_term(kmb, H, D_E_rho0, berry_broadening);

  auto S_E_intrinsic_node_kind = StaticDMKind::S;
  int S_E_intrinsic_impurity_order = 0;
  std::string S_E_intrinsic_name = "S_E^{(0)}"; // TODO intrinsic vs extrinsic in name
  StaticDMGraphNode::ParentsMap S_E_intrinsic_parents {
    {StaticDMDerivedBy::P_inv_DE, std::weak_ptr<StaticDMGraphNode>(eq_node)},
  };
  auto S_E_intrinsic_node = std::make_shared<StaticDMGraphNode>(S_E_intrinsic, S_E_intrinsic_node_kind,
      S_E_intrinsic_impurity_order, S_E_intrinsic_name, S_E_intrinsic_parents);

  eq_node->children[StaticDMDerivedBy::P_inv_DE] = S_E_intrinsic_node;

  // Extrinsic part of S:
  Vec n_E_all = scatter_to_all(n_E);
  auto n_E_all_std = std::get<1>(get_local_contents(n_E_all));
  Mat K_od_n_E = apply_collision_od(kmb, H, sigma, disorder_term_od, n_E_all_std);
  ierr = MatScale(K_od_n_E, -1.0);CHKERRXX(ierr);

  Mat S_E_extrinsic = apply_precession_term(kmb, H, K_od_n_E, berry_broadening);

  auto S_E_extrinsic_node_kind = StaticDMKind::S;
  int S_E_extrinsic_impurity_order = 0;
  std::string S_E_extrinsic_name = "S_E^{(0)}";
  StaticDMGraphNode::ParentsMap S_E_extrinsic_parents {
    {StaticDMDerivedBy::P_inv_Kod, std::weak_ptr<StaticDMGraphNode>(n_E_node)}
  };
  auto S_E_extrinsic_node = std::make_shared<StaticDMGraphNode>(S_E_extrinsic, S_E_extrinsic_node_kind,
      S_E_extrinsic_impurity_order, S_E_extrinsic_name, S_E_extrinsic_parents);

  n_E_node->children[StaticDMDerivedBy::P_inv_Kod] = S_E_extrinsic_node;

  ierr = VecDestroy(&n_E);CHKERRXX(ierr);
  ierr = VecDestroy(&n_E_all);CHKERRXX(ierr);
  ierr = VecDestroy(&D_E_rho0_diag);CHKERRXX(ierr);
  ierr = MatDestroy(&D_E_rho0);CHKERRXX(ierr);
  ierr = MatDestroy(&K_od_n_E);CHKERRXX(ierr);
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
    const kmBasis<k_dim> &kmb, std::array<Mat, k_dim> DH0_cross_Bhat,
    std::array<Mat, k_dim> d_dk_Cart, std::array<Mat, k_dim> R, KSP Kdd_ksp,
    Vec Bhat_dot_Omega, const Hamiltonian &H, const double sigma, const UU_OD &disorder_term_od,
    double berry_broadening) {
  // TODO check that parent_node has the appropriate structure
  // (it should be <rho>_{E, B^{N-1}}).

  // Construct n child: Kdd^{-1} D_B (<rho>).
  Mat D_B_rho = apply_driving_magnetic(kmb, DH0_cross_Bhat, d_dk_Cart, R, parent_node->rho);

  Vec D_B_rho_diag;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm,
      &D_B_rho_diag);CHKERRXX(ierr);
  ierr = MatGetDiagonal(D_B_rho, D_B_rho_diag);CHKERRXX(ierr);

  Vec child_n_B;
  ierr = VecDuplicate(D_B_rho_diag, &child_n_B);CHKERRXX(ierr);
  ierr = KSPSolve(Kdd_ksp, D_B_rho_diag, child_n_B);CHKERRXX(ierr);

  Mat child_n_B_Mat = make_diag_Mat(child_n_B);
  auto child_n_B_node_kind = StaticDMKind::n;
  int child_n_B_impurity_order = parent_node->impurity_order - 1;
  std::string child_n_B_name = ""; // TODO
  StaticDMGraphNode::ParentsMap child_n_B_parents {
    {StaticDMDerivedBy::Kdd_inv_DB, std::weak_ptr<StaticDMGraphNode>(parent_node)}
  };
  auto child_n_B_node = std::make_shared<StaticDMGraphNode>(child_n_B_Mat, child_n_B_node_kind,
      child_n_B_impurity_order, child_n_B_name, child_n_B_parents);

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
  set_Mat_diagonal(D_B_rho, 0.0);
  Mat child_S_B_intrinsic = apply_precession_term(kmb, H, D_B_rho, berry_broadening);

  auto child_S_B_intrinsic_node_kind = StaticDMKind::S;
  int child_S_B_intrinsic_impurity_order = parent_node->impurity_order;
  std::string child_S_B_intrinsic_name = ""; // TODO
  StaticDMGraphNode::ParentsMap child_S_B_intrinsic_parents {
    {StaticDMDerivedBy::P_inv_DB, std::weak_ptr<StaticDMGraphNode>(parent_node)},
  };
  auto child_S_B_intrinsic_node = std::make_shared<StaticDMGraphNode>(child_S_B_intrinsic,
      child_S_B_intrinsic_node_kind, child_S_B_intrinsic_impurity_order,
      child_S_B_intrinsic_name, child_S_B_intrinsic_parents);

  parent_node->children[StaticDMDerivedBy::P_inv_DB] = child_S_B_intrinsic_node;

  // Extrinsic part of S:
  Vec child_n_B_all = scatter_to_all(child_n_B);
  auto child_n_B_all_std = std::get<1>(get_local_contents(child_n_B_all));
  Mat K_od_child_n_B = apply_collision_od(kmb, H, sigma, disorder_term_od, child_n_B_all_std);
  ierr = MatScale(K_od_child_n_B, -1.0);CHKERRXX(ierr);

  Mat child_S_B_extrinsic = apply_precession_term(kmb, H, K_od_child_n_B, berry_broadening);

  auto child_S_B_extrinsic_node_kind = StaticDMKind::S;
  int child_S_B_extrinsic_impurity_order = parent_node->impurity_order;
  std::string child_S_B_extrinsic_name = ""; // TODO
  StaticDMGraphNode::ParentsMap child_S_B_extrinsic_parents {
    {StaticDMDerivedBy::P_inv_Kod, std::weak_ptr<StaticDMGraphNode>(child_n_B_node)},
  };
  auto child_S_B_extrinsic_node = std::make_shared<StaticDMGraphNode>(child_S_B_extrinsic,
      child_S_B_extrinsic_node_kind, child_S_B_extrinsic_impurity_order,
      child_S_B_extrinsic_name, child_S_B_extrinsic_parents);

  child_n_B_node->children[StaticDMDerivedBy::P_inv_Kod] = child_S_B_extrinsic_node;

  if (parent_node->node_kind != StaticDMKind::S) {
    // Construct xi child.
    // xi and n have xi children, but S does not.
    Vec parent_diag;
    ierr = VecDuplicate(D_B_rho_diag, &parent_diag);CHKERRXX(ierr);
    ierr = MatGetDiagonal(parent_node->rho, parent_diag);CHKERRXX(ierr);

    // xi_{km} = n_{km} * (B dot Omega_{km})
    Vec xi;
    ierr = VecDuplicate(D_B_rho_diag, &xi);CHKERRXX(ierr);
    ierr = VecPointwiseMult(xi, parent_diag, Bhat_dot_Omega);CHKERRXX(ierr);

    Mat child_xi_Mat = make_diag_Mat(xi);
    auto child_xi_node_kind = StaticDMKind::xi;
    int child_xi_impurity_order = parent_node->impurity_order;
    std::string child_xi_name = ""; // TODO
    StaticDMGraphNode::ParentsMap child_xi_parents {
      {StaticDMDerivedBy::B_dot_Omega, std::weak_ptr<StaticDMGraphNode>(parent_node)}
    };
    auto child_xi_node = std::make_shared<StaticDMGraphNode>(child_xi_Mat, child_xi_node_kind,
        child_xi_impurity_order, child_xi_name, child_xi_parents);

    parent_node->children[StaticDMDerivedBy::B_dot_Omega] = child_xi_node;

    ierr = VecDestroy(&xi);CHKERRXX(ierr);
  }

  ierr = VecDestroy(&child_n_B);CHKERRXX(ierr);
  ierr = VecDestroy(&child_n_B_all);CHKERRXX(ierr);
  ierr = VecDestroy(&D_B_rho_diag);CHKERRXX(ierr);
  ierr = MatDestroy(&D_B_rho);CHKERRXX(ierr);
  ierr = MatDestroy(&K_od_child_n_B);CHKERRXX(ierr);
}

} // namespace anomtrans

#endif // ANOMTRANS_DM_GRAPH_H
