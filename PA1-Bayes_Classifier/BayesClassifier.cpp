#ifndef Bayes
#define Bayes

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

using namespace Eigen;
using namespace std;

extern float box_muller(float, float);

vector<Vector2f> generateSamples(Vector2f mu, Matrix2f sigma)
{
	vector<Vector2f> samples;
	for(int i = 0; i < 10000; i++)
	{
		samples.push_back(Vector2f(box_muller(mu(0,0), sigma(0,0)), box_muller(mu(1,0), sigma(1,1))));
	}
	
	return samples;
}

int classifierCaseOne(Vector2f x, Vector2f muOne, Vector2f muTwo, float sigmaOne, float sigmaTwo, float priorOne = 0.5, float priorTwo = 0.5)
{
	float discrimOne = -1 * (x-muOne).transpose() * (x - muOne);
	float discrimTwo = -1 * (x - muTwo).transpose() * (x - muTwo);
	
	if(priorOne != priorTwo)
	{
		discrimOne = (discrimOne / (2 * sigmaOne)) + log(priorOne);
		discrimTwo = (discrimTwo / (2 * sigmaTwo)) + log(priorTwo);
	}
	
	if(discrimOne > discrimTwo)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

#endif