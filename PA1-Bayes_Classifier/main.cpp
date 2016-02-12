#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"

using namespace Eigen;
using namespace std;
extern vector<Vector2f> generateSamples(Vector2f mu, Matrix2f sigma);
int main()
{
  Matrix2f stDev;
  Vector2f muOne, muTwo;
  muOne << 1, 1;
  muTwo << 6, 6;
  stDev << 2, 0,
  	   	   0, 2;
  

}