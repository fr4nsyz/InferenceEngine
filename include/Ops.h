#include "Matrix.h"

class Ops {
public:
  virtual Matrix forward(const Matrix &input) const = 0;
};
