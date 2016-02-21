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

int BayesClassifier::classifierCaseTwo(Vector2f x, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo, float priorOne, float priorTwo)
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

int BayesClassifier::classifierCaseThree(Vector2f x, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo, float priorOne,  float priorTwo)
{
	float discrimOne = (x.transpose() * (-0.5 * sigmaOne.inverse()) * x)
						+ ((sigmaOne.inverse() * muOne).transpose() * x)(0)
						+ (-0.5 * muOne.transpose() * sigmaOne.inverse() * muOne)
						+ (-0.5 * log(sigmaOne.determinant()));

	float discrimTwo = (x.transpose() * (-0.5 * sigmaTwo.inverse()) * x)
						+ ((sigmaTwo.inverse() * muTwo).transpose() * x)(0)
						+ (-0.5 * muTwo.transpose() * sigmaTwo.inverse() * muTwo)
						+ (-0.5 * log(sigmaTwo.determinant()));


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

pair<float, float> BayesClassifier::findChernoffBound(Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo)
{
	float chernoffIndex = 0.0;
	float chernoffValue = errorBound(chernoffIndex, muOne, muTwo, sigmaOne, sigmaTwo);
	for(float i = 0.0; i <= 1; i += 0.00001)
	{
		float curChernoffValue = errorBound(i, muOne, muTwo, sigmaOne, sigmaTwo);
		if(curChernoffValue < chernoffValue)
		{
			chernoffIndex = i;
			chernoffValue = curChernoffValue;
		}
	}

	return pair<float, float>(chernoffIndex, chernoffValue);
}

float BayesClassifier::findBhattacharyyaBound(Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo)
{
	return errorBound(0.5, muOne, muTwo, sigmaOne, sigmaTwo);
}

float BayesClassifier::normSquared(Vector2f x)
{
	return x.transpose() * x;
}

float BayesClassifier::errorBound(float beta, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo)
{

	float kb = (beta*(1-beta))/2.0;
	kb *= (muTwo - muOne).transpose() * (beta*sigmaOne + (1-beta)*sigmaTwo).inverse() * (muTwo-muOne);
	kb += 0.5 * log( (beta*sigmaOne + (1-beta)*sigmaTwo).determinant() / (pow(sigmaOne.determinant(), beta) * pow(sigmaTwo.determinant(), 1 - beta)));

	return exp(-1.0 * kb);
}
