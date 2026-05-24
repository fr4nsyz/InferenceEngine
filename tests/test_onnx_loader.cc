#include <gtest/gtest.h>
#include "Model.h"
#include <cmath>
#include <fstream>
#include <vector>

static const std::string fixtures = "tests/fixtures/";

// ── Model 1: Linear(4→3) + ReLU + Linear(3→2) ─────────────
TEST(DemoTest, LinearReLU) {
    Model model(fixtures + "linear_relu.onnx");

    Tensor input({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output = model.run(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_NEAR(output({0, 0}), 4.79f, 1e-4f);
    EXPECT_NEAR(output({0, 1}), 4.49f, 1e-4f);
}

// ── Model 2: Linear(5→4) + ReLU + Linear(4→3) + Softmax ───
TEST(DemoTest, LinearReLUSoftmax) {
    Model model(fixtures + "linear_relu_softmax.onnx");

    Tensor input({1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor output = model.run(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_NEAR(output({0, 0}), 0.333333f, 1e-5f);
    EXPECT_NEAR(output({0, 1}), 0.333333f, 1e-5f);
    EXPECT_NEAR(output({0, 2}), 0.333333f, 1e-5f);

    float sum = output({0, 0}) + output({0, 1}) + output({0, 2});
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ── Model 3: Linear(3→6, no bias) + ReLU + Linear(6→4) + ReLU + Linear(4→2) ──
TEST(DemoTest, DeepMLP) {
    Model model(fixtures + "deep_mlp.onnx");

    Tensor input({1, 3}, {0.5f, 1.5f, 2.5f});
    Tensor output = model.run(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_NEAR(output({0, 0}), 0.608f, 1e-3f);
    EXPECT_NEAR(output({0, 1}), 0.608f, 1e-3f);
}

// ── Model 4: Original simple_relu fixture (Linear(3→2) + ReLU) ──
TEST(DemoTest, SimpleRelu) {
    Model model(fixtures + "simple_relu.onnx");

    Tensor input({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor output = model.run(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_NEAR(output({0, 0}), 3.1f, 1e-5f);
    EXPECT_NEAR(output({0, 1}), 3.1f, 1e-5f);
}
