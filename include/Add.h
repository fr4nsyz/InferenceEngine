#include "../include/Tensor.h"
#include "../include/Ops.h"

class Add : public Ops {
public:
  Add();
  Tensor forward(std::span<Tensor> inputs) const override;
};
