#include "../include/Matrix.h"
#include "../include/Ops.h"

class Add : public Ops {
public:
  Add();
  Matrix forward(std::span<Matrix> inputs) const override;
};
