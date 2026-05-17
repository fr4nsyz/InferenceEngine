#include "../include/Matrix.h"
#include "../include/Ops.h"

class ReLU : public Ops {
public:
  ReLU();
  Matrix forward(const Matrix &input) const override;
};
