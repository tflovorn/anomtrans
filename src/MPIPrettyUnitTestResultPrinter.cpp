// Copied from googletest/src/gtest.cc and modified as appropriate to handle
// testing an MPI process.
// Current implementation: do not print anything unless rank = 0
// (using rank passed during construction).
#include <gtest/gtest.h>
#include "MPIPrettyUnitTestResultPrinter.h"

namespace anomtrans {

  // Fired before each iteration of tests starts.
void MPIPrettyUnitTestResultPrinter::OnTestIterationStart(
    const testing::UnitTest& unit_test, int iteration) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnTestIterationStart(unit_test, iteration);
}

void MPIPrettyUnitTestResultPrinter::OnEnvironmentsSetUpStart(
    const testing::UnitTest& unit_test) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnEnvironmentsSetUpStart(unit_test);
}

void MPIPrettyUnitTestResultPrinter::OnTestCaseStart(const testing::TestCase& test_case) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnTestCaseStart(test_case);
}

void MPIPrettyUnitTestResultPrinter::OnTestStart(const testing::TestInfo& test_info) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnTestStart(test_info);
}

// Called after an assertion failure.
void MPIPrettyUnitTestResultPrinter::OnTestPartResult(
    const testing::TestPartResult& result) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnTestPartResult(result);
}

void MPIPrettyUnitTestResultPrinter::OnTestEnd(const testing::TestInfo& test_info) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnTestEnd(test_info);
}

void MPIPrettyUnitTestResultPrinter::OnTestCaseEnd(const testing::TestCase& test_case) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnTestCaseEnd(test_case);
}

void MPIPrettyUnitTestResultPrinter::OnEnvironmentsTearDownStart(
    const testing::UnitTest& unit_test) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnEnvironmentsTearDownStart(unit_test);
}

// Text printed in Google Test's text output and --gunit_list_tests
// output to label the type parameter and value parameter for a test.
static const char kTypeParamLabel[] = "TypeParam";
static const char kValueParamLabel[] = "GetParam()";

void PrintFullTestCommentIfPresent(const testing::TestInfo& test_info) {
  const char* const type_param = test_info.type_param();
  const char* const value_param = test_info.value_param();

  if (type_param != NULL || value_param != NULL) {
    printf(", where ");
    if (type_param != NULL) {
      printf("%s = %s", kTypeParamLabel, type_param);
      if (value_param != NULL)
        printf(" and ");
    }
    if (value_param != NULL) {
      printf("%s = %s", kValueParamLabel, value_param);
    }
  }
}

// Internal helper for printing the list of failed tests.
void MPIPrettyUnitTestResultPrinter::PrintFailedTests(const testing::UnitTest& unit_test) {
  // TODO better way to handle this being a static function?
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  const int failed_test_count = unit_test.failed_test_count();
  if (failed_test_count == 0) {
    return;
  }

  for (int i = 0; i < unit_test.total_test_case_count(); ++i) {
    const testing::TestCase& test_case = *unit_test.GetTestCase(i);
    if (!test_case.should_run() || (test_case.failed_test_count() == 0)) {
      continue;
    }
    for (int j = 0; j < test_case.total_test_count(); ++j) {
      const testing::TestInfo& test_info = *test_case.GetTestInfo(j);
      if (!test_info.should_run() || test_info.result()->Passed()) {
        continue;
      }
      printf("[  FAILED  ] ");
      printf("%s.%s", test_case.name(), test_info.name());
      PrintFullTestCommentIfPresent(test_info);
      printf("\n");
    }
  }
}

void MPIPrettyUnitTestResultPrinter::OnTestIterationEnd(const testing::UnitTest& unit_test,
                                                     int iteration) {
  if (rank != 0) {
    return;
  }
  default_result_printer->OnTestIterationEnd(unit_test, iteration);
}

} // namespace anomtrans
