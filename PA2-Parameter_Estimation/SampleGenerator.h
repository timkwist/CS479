#ifndef SAMPLEGENERATOR_H
#define SAMPLEGENERATOR_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

using namespace Eigen;
using namespace std;

extern float box_muller(float, float);

class SampleGenerator
{
public:
	/**
	 * Generate 10,000 Gaussian random samples given specifc mu and sigma
	 * 
	 * @param mu 2x1 vector
	 * @param sigma 2x2 matrix
	 * 
	 * @return Vector of size 10,000, of 2x1 vectors
	 */
	vector<Vector2f> generateSamples(Vector2f mu, Matrix2f sigma);

};

#endif