#include "../include/Matrix.h"
#include "../include/Ops.h"

class Linear : public Ops {
  // Note the following are const because they are initialized with their proper
  // values on model loading
  const Matrix _weights;
  const Matrix _bias; // Vector, but use Matrix class
  int _in_feats;
  int _out_feats;

public:
  Linear(int in_feats, int out_feats, const Matrix &weights,
         const Matrix &bias);
  Matrix forward(std::span<Matrix> inputs) const override;
};
