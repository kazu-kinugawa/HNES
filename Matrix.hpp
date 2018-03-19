#pragma once

#include <Eigen/Core>

#define USE_FLOAT

#ifdef USE_FLOAT
typedef float Real;
typedef Eigen::MatrixXf MatD;
typedef Eigen::VectorXf VecD;
#else
typedef double Real;
typedef Eigen::MatrixXd MatD;
typedef Eigen::VectorXd VecD;
#endif

typedef Eigen::Vector3d Vec3D;
typedef Eigen::MatrixXi MatI;
typedef Eigen::VectorXi VecI;
#define REAL_MAX std::numeric_limits<Real>::max()

enum MODE{
  TRAIN, TEST,
};

enum CHECK{
  OFF,// run on server
  ONE_EPOCH_CHECK,// run on server with tiny parameters
  GRAD_CHECK,// run on local
};
