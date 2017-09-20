#include "dm_graph.h"

namespace anomtrans {

std::shared_ptr<DMGraphNode> make_eq_node(Vec Ekm, double beta, double mu) {
    Vec rho0_km = make_rho0(Ekm, beta, mu);
    Mat rho0_km_Mat = make_diag_Mat(rho0_km);

    DMKind node_kind = DMKind::n;
    int impurity_order = 0;
    std::string name = "\\rho_0";
    DMGraphNode::ParentsMap parents;

    auto rho0_node = std::make_shared<DMGraphNode>(rho0_km_Mat, node_kind, impurity_order, name, parents);

    PetscErrorCode ierr = VecDestroy(&rho0_km);CHKERRXX(ierr);

    return rho0_node;
}

} // namespace anomtrans
