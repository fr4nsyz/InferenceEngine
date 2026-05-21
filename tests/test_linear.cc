#include "../include/Linear.h"
#include <gtest/gtest.h>

TEST(LinearTest, ForwardShape) {
  Tensor weights({3, 2});
  weights({0, 0}) = 1; weights({0, 1}) = 2;
  weights({1, 0}) = 3; weights({1, 1}) = 4;
  weights({2, 0}) = 5; weights({2, 1}) = 6;

  Tensor bias({3, 1});
  bias({0, 0}) = 1;
  bias({1, 0}) = 2;
  bias({2, 0}) = 3;

  Linear lin(2, 3, weights, bias);

  Tensor input({2, 1});
  input({0, 0}) = 2;
  input({1, 0}) = 3;

  Tensor out = lin.forward(input);
  EXPECT_EQ(out.shape(), std::vector<int>({3, 1}));
}

TEST(LinearTest, ForwardValues) {
  Tensor weights({2, 2});
  weights({0, 0}) = 1; weights({0, 1}) = 2;
  weights({1, 0}) = 3; weights({1, 1}) = 4;

  Tensor bias({2, 1});
  bias({0, 0}) = 1;
  bias({1, 0}) = 2;

  Linear lin(2, 2, weights, bias);

  Tensor input({2, 1});
  input({0, 0}) = 2;
  input({1, 0}) = 3;

  Tensor out = lin.forward(input);
  EXPECT_FLOAT_EQ(out({0, 0}), 1*2 + 2*3 + 1);
  EXPECT_FLOAT_EQ(out({1, 0}), 3*2 + 4*3 + 2);
}
