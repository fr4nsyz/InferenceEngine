#include "Matrix.h"
#include "Ops.h"

class Softmax : public Ops {
public:
  Softmax();
  Matrix forward(const Matrix &input) const override;
};
