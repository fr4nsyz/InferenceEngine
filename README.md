# InferenceEngine

A C++20 inference engine that loads and runs ONNX models. Uses protobuf to parse `.onnx` files and translates ONNX ops into native C++ operations at load time. No graph IR (Intermediate Representation) or runtime dispatch needed for sequential models.

## Architecture

```
.onnx file ──► ONNXLoader (protobuf parse + op translation) ──► vector<unique_ptr<Ops>> ──► Model::run()
```

**ONNXLoader** reads the `.onnx` protobuf, extracts initializer tensors (weights/biases), and translates each ONNX node into a native C++ op object. This translation happens once at load time.

| ONNX Op    | Native Op   | What happens at load time                                    |
|------------|-------------|--------------------------------------------------------------|
| `Gemm`     | `Linear`    | Weight transposed if `transB=1`, bias extracted, `Linear` constructed |
| `Relu`     | `ReLU`      | Direct construction                                          |
| `Softmax`  | `Softmax`   | Axis attribute read, `Softmax(axis)` constructed             |
| `Add`      | `Add`       | Direct construction                                          |

**Model::run()** threads a single `Tensor` through the op list sequentially. Input goes into the first op, its output feeds the next, and so on.

## Project Structure

```
include/
  Tensor.h        Dense float tensor with matmul, broadcast add, transpose, reshape
  Ops.h           Abstract op base class (virtual forward)
  Linear.h        Linear layer: input @ weights + bias
  ReLU.h          Element-wise max(0, x)
  Softmax.h       Softmax along an axis
  Add.h           Element-wise addition with broadcasting
  ONNXLoader.h    Static load() → vector<unique_ptr<Ops>>
  Model.h         Facade: construct from .onnx path, run inference
src/
  Tensor.cc       Tensor implementation (2D matmul, broadcasting, transpose)
  Ops.cc          Virtual destructor
  Linear.cc       input * weights + bias
  ReLU.cc         max(0, x)
  Softmax.cc      exp(x) / sum(exp(x)) along axis
  Add.cc          Broadcast addition
  ONNXLoader.cc   Protobuf deserialization + op translation
  Model.cc        Sequential execution loop
tests/
  test_tensor.cc       Tensor construction, indexing, arithmetic
  test_linear.cc       Linear forward pass (shape + values)
  test_relu.cc         ReLU forward
  test_softmax.cc      Softmax sum-to-one, range, ordering
  test_add.cc          Broadcast addition
  test_onnx_loader.cc  End-to-end ONNX load + inference (4 models)
  generate_fixtures.py Export PyTorch models to ONNX + golden values
  fixtures/            Pre-built .onnx models + metadata JSONs
```

## Dependencies

Fetched automatically via CMake FetchContent:

- **protobuf v28.3** (afaik is the minimum version that supported C++20 support + working FetchContent CMake) -> ONNX file deserialization
- **abseil-cpp** -> required by protobuf
- **GoogleTest v1.15.2** -> test framework (tests subdirectory only)

The ONNX proto schema (`onnx.proto`) is downloaded at configure time from the official ONNX repository.

## Build

Requires CMake 3.14+, a C++20 compiler, and an internet connection for FetchContent.

```bash
cmake -B build
cmake --build build
```

To build without tests:

```bash
cmake -B build -DBUILD_TESTS=OFF
cmake --build build
```

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

This runs 6 test binaries covering all ops and ONNX loading.

## Demo: ONNX Model Inference

The `test_onnx_loader` binary is the end-to-end demo. It loads real ONNX models exported from PyTorch and verifies the inference output matches PyTorch's results.

### What's being tested

| Model | Architecture | Input Shape | ONNX Ops |
|-------|-------------|-------------|----------|
| `simple_relu.onnx` | Linear(3→2) + ReLU | `[1, 3]` | Gemm, Relu |
| `linear_relu.onnx` | Linear(4→3) + ReLU + Linear(3→2) | `[1, 4]` | Gemm, Relu |
| `linear_relu_softmax.onnx` | Linear(5→4) + ReLU + Linear(4→3) + Softmax | `[1, 5]` | Gemm, Relu, Softmax |
| `deep_mlp.onnx` | Linear(3→6) + ReLU + Linear(6→4) + ReLU + Linear(4→2) | `[1, 3]` | Gemm, Relu |

### Run the demo

```bash
./build/tests/test_onnx_loader
```

Expected output:

```
[==========] Running 4 tests from 1 test suite.
[ RUN      ] DemoTest.LinearReLU
[       OK ] DemoTest.LinearReLU
[ RUN      ] DemoTest.LinearReLUSoftmax
[       OK ] DemoTest.LinearReLUSoftmax
[ RUN      ] DemoTest.DeepMLP
[       OK ] DemoTest.DeepMLP
[ RUN      ] DemoTest.SimpleRelu
[       OK ] DemoTest.SimpleRelu
[==========] 4 tests ran.
[  PASSED  ] 4 tests.
```

### Regenerate the ONNX fixtures

If you want to modify the test models or add new ones, you need `torch` and `onnx` installed in your Python environment:

```bash
pip install torch onnx
python3 tests/generate_fixtures.py
```

This writes `.onnx` files and `_meta.json` files (containing expected input/output values) into `tests/fixtures/`. To add a new model:

1. Define it in `generate_fixtures.py` with deterministic weights
2. Call `export_model()` with a known input tensor
3. Add a matching `TEST_F` case in `test_onnx_loader.cc` using the golden values from the JSON

## Current Limitations

- **Sequential models only** -> no branch/merge/skip connections (flat `vector<Ops>` pipeline)
- **Single input, single output** -> `Model::run()` takes one `Tensor` and returns one `Tensor`
- **Supported ONNX ops**: Gemm, Relu, Softmax, Add
- **Float32 only** -> no int8, float16, or other data types
- **2D matmul only** -> Tensor matrix multiply requires both operands to be 2D

## Usage Example

```cpp
#include "Model.h"

int main() {
    Model model("my_model.onnx");
    Tensor input({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output = model.run(input);
    // output.shape() == {1, 2}
}
```
