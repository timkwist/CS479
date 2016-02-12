#include <iostream>
#include <Eigen/Dense>
#include <vector>
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
  vector<Vector2f> sampleOne = generateSamples(muOne, stDev),
  					sampleTwo = generateSamples(muTwo, stDev);

  					cout << sampleOne[0];
  

}