#include "../include/Matrix.h"

class ReLU {
public:
  ReLU();
  Matrix forward(const Matrix &input) const;
};
