// Trilinos unit testing framework standard main().
// Discussed at:
// https://trilinos.org/docs/dev/packages/teuchos/doc/html/group__Teuchos__UnitTest__grp.html
// https://trilinos.org/pipermail/trilinos-users/2014-August/004225.html

#include <Teuchos_UnitTestRepository.hpp>
#include <Teuchos_GlobalMPISession.hpp>

int main(int argc, char* argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
}
