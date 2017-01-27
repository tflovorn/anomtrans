#include "mpi.h"

namespace anomtrans {

MPIComm get_comm() {
  return Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
}

} // namespace anomtrans
