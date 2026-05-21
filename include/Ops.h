#include "Tensor.h"
#include <span>

class Ops {
public:
  virtual ~Ops();
  virtual Tensor forward(std::span<Tensor> inputs) const = 0;
};
