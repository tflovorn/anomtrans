#ifndef ANOMTRANS_DM_GRAPH_H
#define ANOMTRANS_DM_GRAPH_H

#include <string>
#include <map>
#include <memory>
#include <petscksp.h>
#include "mat.h"
#include "grid_basis.h"
#include "rho0.h"
#include "driving.h"

namespace anomtrans {

/** @brief Kind of density matrix contribution: n = diagonal (n), S = off-diagonal,
 *         xi = diagonal via P^{-1} D_B ~ B dot Omega.
 */
enum struct DMKind { n, S, xi };

/** @brief Identifier for the operation used to derive a child node from a parent node.
 */
enum struct DMDerivedBy { P_inv_DB, B_dot_Omega, Kdd_inv_DE, P_inv_DE, P_inv_Kod, Kdd_inv_DB };

class DMGraphNode {
public:
  using ParentsMap = std::map<DMDerivedBy, std::weak_ptr<DMGraphNode>>;
  using ChildrenMap = std::map<DMDerivedBy, std::shared_ptr<DMGraphNode>>;

  /** @brief Density matrix given by this node. This matrix is owned by this node,
   *         i.e. the node is responsible for destroying it.
   *  @todo Would like to expose only const Mat, but many PETSc functions which are logically
   *        const do not take const Mat (e.g. MatMult). Could protect this value by implementing
   *        pass-through methods for required PETSc functions.
   */
  Mat rho;

  const DMKind node_kind;

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
  DMGraphNode(Mat _rho, DMKind _node_kind, int _impurity_order, std::string _name,
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
  std::map<std::string, std::shared_ptr<DMGraphNode>> unique_descendants();
};

/** @brief Construct the node containing the equilibrium density matrix
 *         (which has no parent nodes).
 */
DMGraphNode make_eq_node(Vec Ekm, double beta, double mu);

/** @brief Given a node containing the equilibrium density matrix,
 *         add children to it corresponding to linear response to an applied magnetic
 *         field and return a list of those children.
 *  @todo Add parameters necessary to implement this.
 *  @todo Implementation can check that rho0 is really the equilibrium density matrix
 *        by verifying that its list of parents is empty.
 */
DMGraphNode::ChildrenMap add_linear_response_magnetic(DMGraphNode &eq_node);

/** @brief Given a node containing the equilibrium density matrix,
 *         add children to it corresponding to linear response to an applied electric
 *         field and return a list of those children.
 *  @todo Add parameters necessary to implement this.
 *  @todo Implementation can check that rho0 is really the equilibrium density matrix
 *        by verifying that its list of parents is empty.
 *  @precondition Kdd_ksp should have its nullspace set appropriately before this function is
 *                called (the nullspace of Kdd is the <rho_0> vector).
 */
template <std::size_t k_dim>
DMGraphNode::ChildrenMap add_linear_response_electric(std::shared_ptr<DMGraphNode> eq_node,
    const kmBasis<k_dim> &kmb, Mat Ehat_dot_grad_k, Mat Ehat_dot_R, KSP Kdd_ksp) {
  // Construct <n_E^(-1)> = Kdd^{-1} D_E (<rho_0>).
  Mat D_E_rho0_Mat = apply_driving_electric(kmb, Ehat_dot_grad_k, Ehat_dot_R, eq_node->rho);

  Vec D_E_rho0;
  PetscErrorCode ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kmb.end_ikm,
      &D_E_rho0);CHKERRXX(ierr);
  ierr = MatGetDiagonal(D_E_rho0_Mat, D_E_rho0);CHKERRXX(ierr);

  Vec n_E;
  ierr = VecDuplicate(D_E_rho0, &n_E);CHKERRXX(ierr);
  ierr = KSPSolve(Kdd_ksp, D_E_rho0, n_E);CHKERRXX(ierr);

  Mat n_E_Mat = make_diag_Mat(n_E);
  DMKind n_E_node_kind = DMKind::n;
  int n_E_impurity_order = -1;
  std::string n_E_name = "n_E^{(-1)}";
  DMGraphNode::ParentsMap n_E_parents {
    {DMDerivedBy::Kdd_inv_DE, std::weak_ptr<DMGraphNode>(eq_node)}
  };
  auto n_E_node = std::make_shared<DMGraphNode>(n_E_Mat, n_E_node_kind, n_E_impurity_order,
      n_E_name, n_E_parents);

  eq_node->children[DMDerivedBy::Kdd_inv_DE] = n_E_node;

  // TODO: Construct <S_E^(0)>.

  ierr = VecDestroy(&n_E);CHKERRXX(ierr);
  ierr = VecDestroy(&D_E_rho0);CHKERRXX(ierr);
  ierr = MatDestroy(&D_E_rho0_Mat);CHKERRXX(ierr);

  return eq_node->children;
}

/** @brief Add children to `node` corresponding to the next order of the expansion in powers
 *         of magnetic field and return a list of those children.
 *  @precondition `node` should already include the electric field linear response,
 *                i.e. it should be <n_E^{(-1)}>, <S_E^{(0)}>, or one of their descendants.
 */
DMGraphNode::ChildrenMap add_next_order_magnetic(DMGraphNode &parent_node);

} // namespace anomtrans

#endif // ANOMTRANS_DM_GRAPH_H
