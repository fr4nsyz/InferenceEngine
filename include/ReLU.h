#include "../include/Tensor.h"
#include "../include/Ops.h"

class ReLU : public Ops {
public:
  ReLU();
  Tensor forward(std::span<Tensor> inputs) const override;
};
