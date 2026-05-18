#include "Matrix.h"
#include <span>

class Ops {
public:
  virtual ~Ops(); // override the ops destructor because we want to make sure
                  // memory is cleared where appropriate for variable input
                  // sizes
  virtual Matrix forward(std::span<Matrix> inputs) const = 0;
};
