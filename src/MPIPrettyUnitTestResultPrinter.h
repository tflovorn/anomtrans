#ifndef ANOMTRANS_MPIPRETTYUNITTESTRESULTPRINTER_H
#define ANOMTRANS_MPIPRETTYUNITTESTRESULTPRINTER_H

#include <cstdio>
#include <mpi.h>
#include <petscksp.h>
#include <gtest/gtest.h>

namespace anomtrans {

class MPIPrettyUnitTestResultPrinter : public testing::TestEventListener {
 public:
  MPIPrettyUnitTestResultPrinter(testing::TestEventListener *_default_result_printer, int _rank) :
      default_result_printer(_default_result_printer), rank(_rank) {}
  /*
  ~MPIPrettyUnitTestResultPrinter() {
    delete default_result_printer;
  }
  */
  static void PrintTestName(const char * test_case, const char * test) {
    // TODO is there a better way to do this?
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) {
      return;
    }
    printf("%s.%s", test_case, test);
  }

  // The following methods override what's in the TestEventListener class.
  virtual void OnTestProgramStart(const testing::UnitTest& /*unit_test*/) {}
  virtual void OnTestIterationStart(const testing::UnitTest& unit_test, int iteration);
  virtual void OnEnvironmentsSetUpStart(const testing::UnitTest& unit_test);
  virtual void OnEnvironmentsSetUpEnd(const testing::UnitTest& /*unit_test*/) {}
  virtual void OnTestCaseStart(const testing::TestCase& test_case);
  virtual void OnTestStart(const testing::TestInfo& test_info);
  virtual void OnTestPartResult(const testing::TestPartResult& result);
  virtual void OnTestEnd(const testing::TestInfo& test_info);
  virtual void OnTestCaseEnd(const testing::TestCase& test_case);
  virtual void OnEnvironmentsTearDownStart(const testing::UnitTest& unit_test);
  virtual void OnEnvironmentsTearDownEnd(const testing::UnitTest& /*unit_test*/) {}
  virtual void OnTestIterationEnd(const testing::UnitTest& unit_test, int iteration);
  virtual void OnTestProgramEnd(const testing::UnitTest& /*unit_test*/) {}

 private:
  static void PrintFailedTests(const testing::UnitTest& unit_test);

  testing::TestEventListener *default_result_printer;
  const int rank;
};

} // namespace anomtrans

#endif // ANOMTRANS_MPIPRETTYUNITTESTRESULTPRINTER_H
