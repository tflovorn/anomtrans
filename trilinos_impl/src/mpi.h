#ifndef ANOMTRANS_MPI_H
#define ANOMTRANS_MPI_H

#include <Tpetra_DefaultPlatform.hpp>
#include <Teuchos_GlobalMPISession.hpp>

namespace anomtrans {

using MPIComm = Teuchos::RCP<const Teuchos::Comm<int>>;

/** @brief Get the Tpetra default communicator. This should not be called until
 *         MPI has been set up by creating a Teuchos::GlobalMPISession.
 */
MPIComm get_comm();

} // namespace anomtrans

#endif // ANOMTRANS_MPI_H
