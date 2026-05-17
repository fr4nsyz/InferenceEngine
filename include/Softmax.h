#include "Matrix.h"

class Softmax {
public:
  Softmax();
  Matrix forward(const Matrix &input) const;
};
