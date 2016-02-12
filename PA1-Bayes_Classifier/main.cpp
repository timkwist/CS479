#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"

using namespace Eigen;
using namespace std;

int main()
{
	srand(time(NULL));
	Matrix2f sigma;
	Vector2f muOne, muTwo;
	BayesClassifier classifier;
	vector<Vector2f> sampleOne, sampleTwo;
	int misclassifiedOne = 0, misclassifiedTwo = 0;
	muOne << 1, 1;
	muTwo << 6, 6;
	sigma << 2, 0,
	0, 2;
	sampleOne = classifier.generateSamples(muOne, sigma);
	sampleTwo = classifier.generateSamples(muTwo, sigma);

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseOne(sampleOne[i], muOne, muTwo, sigma(0,0), sigma(0,0)) == 2)
		{
			misclassifiedOne++;
		}
		if(classifier.classifierCaseOne(sampleTwo[i], muOne, muTwo, sigma(0,0), sigma(0,0)) == 1)
		{
			misclassifiedTwo++;
		}
	}
	cout << "Samples from one misclassified: " << misclassifiedOne << "\n";
	cout << "Samples from two misclassified: " << misclassifiedTwo << "\n";
	cout << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << "\n";


}