#include "NNSE.hpp"

int main(){
  Eigen::initParallel();// for multi thread

  // NNSEGradChecker::test();

  const CHECK check = OFF;
  // const CHECK check = ONE_EPOCH_CHECK;

  NNSE::train(check);
  // NNSE::test(check);
}
