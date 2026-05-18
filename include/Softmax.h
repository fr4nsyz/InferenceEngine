#include "Matrix.h"
#include "Ops.h"

class Softmax : public Ops {
public:
  Softmax();
  Matrix forward(std::span<Matrix> inputs) const override;
};
