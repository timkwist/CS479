#ifndef MLE_H
#define MLE_H

#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

class MLE
{
public:
	/**
	 * Calculates the sample mean of the given data by summing up the data and dividing
	 * by the total number of samples.
	 * 
	 * @param data vector of 2x1 input samples
	 * @return 2x1 sample mean vector
	 */
	static Vector2f calculateSampleMean(vector<Vector2f> data);

	/**
	 * Calculates the sample covariance of the given data by summing up the difference
	 * between the square of each sample and the sample mean, then dividing the entire sum
	 * by the total number of samples.
	 * 
	 * @param data vector of 2x1 input samples
	 * @param sampleMean 2x1 sample mean
	 * @return 2x2 sample covariance matrix
	 */
	static Matrix2f calculateSampleCovariance(vector<Vector2f> data, Vector2f sampleMean);
};

#endif