#include <iostream>
#include <gtest/gtest.h>


TEST(MainTestSuite, TestName) {
	EXPECT_TRUE(false);
	EXPECT_TRUE(false) << "Error msg";
	ASSERT_TRUE(false);
	EXPECT_TRUE(false);
}
