#include "../include/Tensor.h"
#include "../include/Ops.h"

class Softmax : public Ops {
  int _axis;
public:
  Softmax(int axis = -1);
  Tensor forward(std::span<Tensor> inputs) const override;
};
