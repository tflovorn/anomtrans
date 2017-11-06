#ifndef ANOMTRANS_DYN_DM_GRAPH_H
#define ANOMTRANS_DYN_DM_GRAPH_H

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

/** @brief Kind of density matrix contribution: n = diagonal (n), S = off-diagonal.
 */
enum struct DynDMKind { n, S };

/** @brief Identifier for the operation used to derive a child node from a parent node.
 */
enum struct DynDMDerivedBy { omega_inv_DE, P_inv_DE, Kdd_inv_DE, P_inv_Kod };

/** @brief Graph node for response to a dynamic electric field (high frequency
 *         regime, omega >> 1/relaxation time).
 */
using DynDMGraphNode = DMGraphNode<DynDMKind, DynDMDerivedBy>;

} // namespace anomtrans

#endif // ANOMTRANS_DYN_DM_GRAPH_H
