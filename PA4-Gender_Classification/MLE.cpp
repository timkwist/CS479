#include "MLE.h"
#include <iostream>

Vector2f MLE::calculateSampleMean(vector<Vector2f> data)
{
	Vector2f sum;
	sum << 0.0, 0.0;
	for(vector<int>::size_type i = 0; i < data.size(); i++)
	{
		sum += data[i];
	}
	return sum / data.size();
}

Matrix2f MLE::calculateSampleCovariance(vector<Vector2f> data, Vector2f sampleMean)
{
	Matrix2f sum;
	sum << 	0.0, 0.0,
			0.0, 0.0;
	for(vector<int>::size_type i = 0; i <  data.size(); i++)
	{
		sum += (sampleMean - data[i])*((sampleMean -data[i]).transpose());
	}
	return sum / data.size();
}

VectorXf MLE::calculateSampleMean(vector<VectorXf> data)
{
	VectorXf sum;

	if(data.size() <= 0)
		return sum;

	sum = VectorXf::Zero( data[0].rows() );

	for(vector<int>::size_type i = 0; i < data.size(); i++)
	{
		sum += data[i];
	}
	return sum / data.size();
}

MatrixXf MLE::calculateSampleCovariance(vector<VectorXf> data, VectorXf sampleMean)
{
	MatrixXf coVar;

	if(data.size() <= 0)
		return coVar;

	coVar = MatrixXf::Zero( data[0].rows(), data[0].rows());

	for(vector<int>::size_type i = 0; i <  data.size(); i++)
	{
		coVar += (sampleMean - data[i])*((sampleMean -data[i]).transpose());
	}
	return coVar / data.size();
}

