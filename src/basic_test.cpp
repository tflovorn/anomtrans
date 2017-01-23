#include <gtest/gtest.h>
#include <boost/mpi.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  mpi::environment env(argc, argv);
  return RUN_ALL_TESTS();
}

TEST(BasicTest, Basic) {
  mpi::communicator world;

  EXPECT_EQ(1, 1);
}
