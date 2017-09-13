#ifndef ANOMTRANS_DM_GRAPH_H
#define ANOMTRANS_DM_GRAPH_H

#include <vector>
#include <memory>
#include <petscksp.h>

namespace anomtrans {

/** @brief Type of density matrix contribution: n = diagonal (n), S = off-diagonal,
 *         xi = diagonal via P^{-1} D_B ~ B dot Omega.
 */
enum struct DMType { n, S, xi };

/** @brief Identifier for the operation used to derive a child node from a parent node.
 */
enum struct DMDerivedBy { P_inv_DB, B_dot_Omega, Kdd_inv_DE, P_inv_DE, P_inv_Kod, Kdd_inv_DB };

class DMGraphNode {
public:
  using ParentsList = std::vector<std::pair<std::weak_ptr<DMGraphNode>, DMDerivedBy>>;
  using ChildrenList = std::vector<std::pair<std::shared_ptr<DMGraphNode>, DMDerivedBy>>;

  /** @brief Density matrix given by this node. This matrix is owned by this node,
   *         i.e. the node is responsible for destroying it.
   *  @todo Would like to expose only const Mat, but many PETSc functions which are logically
   *        const do not take const Mat (e.g. MatMult). Could protect this value by implementing
   *        pass-through methods for required PETSc functions.
   */
  Mat rho;

  const DMType node_type;
  const ParentsList parents;
  ChildrenList children;

  /** @note When this node is constructed, it takes ownership of the matrix rho.
   *  @note The vector of children is default-initialized, i.e. it starts empty.
   */
  DMGraphNode(Mat _rho, DMType _node_type, ParentsList _parents)
      : rho(_rho), node_type(_node_type), parents(_parents) {}

  ~DMGraphNode() {
    PetscErrorCode ierr = MatDestroy(&rho);CHKERRXX(ierr);
    // TODO other cleanup required?
  }

  /** @brief Returns a list of all descendants of this node, without duplicates.
   */
  std::vector<std::shared_ptr<DMGraphNode>> unique_descendants();
};

/** @brief Given a node containing the equilibrium density matrix,
 *         add children to it corresponding to linear response to an applied magnetic
 *         field and return a list of those children.
 *  @todo Add parameters necessary to implement this.
 *  @todo Implementation can check that rho0 is really the equilibrium density matrix
 *        by verifying that its list of parents is empty.
 */
DMGraphNode::ChildrenList add_linear_response_magnetic(DMGraphNode eq_node);

/** @brief Given a node containing the equilibrium density matrix,
 *         add children to it corresponding to linear response to an applied electric
 *         field and return a list of those children.
 *  @todo Add parameters necessary to implement this.
 *  @todo Implementation can check that rho0 is really the equilibrium density matrix
 *        by verifying that its list of parents is empty.
 */
DMGraphNode::ChildrenList add_linear_response_electric(DMGraphNode eq_node);

/** @brief Add children to `node` corresponding to the next order of the expansion in powers
 *         of magnetic field and return a list of those children.
 *  @precondition `node` should already include the electric field linear response,
 *                i.e. it should be <n_E^{(-1)}>, <S_E^{(0)}>, or one of their descendants.
 */
DMGraphNode::ChildrenList add_next_order_magnetic(DMGraphNode node);

} // namespace anomtrans

#endif // ANOMTRANS_DM_GRAPH_H
