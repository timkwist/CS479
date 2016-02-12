#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

using namespace Eigen;
using namespace std;

extern float box_muller(float, float);

class BayesClassifier
{
public:
	vector<Vector2f> generateSamples(Vector2f mu, Matrix2f sigma);
	int classifierCaseOne(Vector2f x, Vector2f muOne, Vector2f muTwo, float varianceOne, float varianceTwo, float priorOne = 0.5, float priorTwo = 0.5);
private:
	float normSquared(Vector2f x);
};

#endif