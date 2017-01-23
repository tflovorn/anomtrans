#include <gtest/gtest.h>
#include <petscksp.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  
  int test_result = RUN_ALL_TESTS();
  
  int ierr = PetscFinalize();

  return test_result;
}

TEST(BasicTest, Basic) {
  EXPECT_EQ(1, 1);
}
