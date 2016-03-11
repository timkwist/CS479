#include "MLE.h"

Vector2f MLE::calculateSampleMean(vector<Vector2f> data)
{
	Vector2f sum;
	for(int i = 0; i < data.size(); i++)
	{
		sum += data[i];
	}
	return sum / data.size();
}

Matrix2f MLE::calculateSampleCovariance(vector<Vector2f> data, Vector2f sampleMean)
{
	Matrix2f sum;
	for(int i = 0; i <  data.size(); i++)
	{
		sum += (sampleMean - data[i])*(sampleMean -data[i]).transpose();
	}
	return sum / data.size();
}