#include "BayesClassifier.h"

vector<Vector2f> BayesClassifier::generateSamples(Vector2f mu, Matrix2f sigma)
{
	vector<Vector2f> samples;
	for(int i = 0; i < 10000; i++)
	{
		samples.push_back(Vector2f(box_muller(mu(0,0), sigma(0,0)), box_muller(mu(1,0), sigma(1,1))));
	}
	
	return samples;
}

int BayesClassifier::classifierCaseOne(Vector2f x, Vector2f muOne, Vector2f muTwo, float varianceOne, float varianceTwo, float priorOne, float priorTwo)
{
	float discrimOne = (((1.0/varianceOne) * muOne).transpose() * x) - (1.0/(2*varianceOne)) * normSquared(muOne);
	float discrimTwo = (((1.0/varianceTwo) * muTwo).transpose() * x) - (1.0/(2*varianceTwo)) * normSquared(muTwo);
	
	if(priorOne != priorTwo)
	{
		discrimOne += log(priorOne);
		discrimTwo += log(priorTwo);
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

int BayesClassifier::classifierCaseTwo(Vector2f x, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo, float priorOne,  float priorTwo)
{
	float discrimOne = ((sigmaOne.inverse() * muOne).transpose() * x)(0) - (0.5 * muOne.transpose() * sigmaOne.inverse() * muOne);
	float discrimTwo = ((sigmaTwo.inverse() * muTwo).transpose() * x)(0) - (0.5 * muTwo.transpose() * sigmaTwo.inverse() * muTwo);

	if(priorOne != priorTwo)
	{
		discrimOne += log(priorOne);
		discrimTwo += log(priorTwo);
	}

	if(discrimOne > discrimTwo)
		return 1;
	else
		return 2;


}

int BayesClassifier::minimumDistanceClassifier(Vector2f x, Vector2f muOne, Vector2f muTwo)
{
	float discrimOne = -1.0 * normSquared(x-muOne);
	float discrimTwo = -1.0 * normSquared(x-muTwo);
	
	if(discrimOne > discrimTwo)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

float BayesClassifier::normSquared(Vector2f x)
{
	return x.transpose() * x;
}
