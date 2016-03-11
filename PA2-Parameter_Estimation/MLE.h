#ifndef MLE_H
#define MLE_H

#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

class MLE
{
public:
	static Vector2f calculateSampleMean(vector<Vector2f> data);

	static Matrix2f calculateSampleCovariance(vector<Vector2f> data, Vector2f sampleMean);
};

#endif