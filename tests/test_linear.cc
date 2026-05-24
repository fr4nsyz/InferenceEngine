#include "../include/Linear.h"
#include <gtest/gtest.h>

TEST(LinearTest, ForwardShape) {
    Tensor weights({3, 2});
    weights({0, 0}) = 1; weights({0, 1}) = 2;
    weights({1, 0}) = 3; weights({1, 1}) = 4;
    weights({2, 0}) = 5; weights({2, 1}) = 6;

    Tensor bias({1, 2});
    bias({0, 0}) = 1; bias({0, 1}) = 2;

    Linear lin(3, 2, weights, bias);

    Tensor input({1, 3});
    input({0, 0}) = 2; input({0, 1}) = 3; input({0, 2}) = 4;

    Tensor arr[] = {input};
    Tensor out = lin.forward(arr);
    EXPECT_EQ(out.shape(), std::vector<int>({1, 2}));
}

TEST(LinearTest, ForwardValues) {
    Tensor weights({2, 2});
    weights({0, 0}) = 1; weights({0, 1}) = 2;
    weights({1, 0}) = 3; weights({1, 1}) = 4;

    Tensor bias({1, 2});
    bias({0, 0}) = 1; bias({0, 1}) = 2;

    Linear lin(2, 2, weights, bias);

    Tensor input({1, 2});
    input({0, 0}) = 2; input({0, 1}) = 3;

    Tensor arr[] = {input};
    Tensor out = lin.forward(arr);
    EXPECT_FLOAT_EQ(out({0, 0}), 1*2 + 3*3 + 1);
    EXPECT_FLOAT_EQ(out({0, 1}), 2*2 + 4*3 + 2);
}
