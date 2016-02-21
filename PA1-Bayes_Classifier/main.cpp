#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"

#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

void writeSamplesToFile(const char* fileName, vector<Vector2f> sampleOne, vector<Vector2f> sampleTwo);

float errorBound(float beta, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo)
{
	float kb = (beta*(1-beta))/2.0;
	kb *= (muTwo - muOne).transpose() * (beta*sigmaOne + (1-beta)*sigmaTwo).inverse() * (muTwo-muOne);
	kb += 0.5 * log( (beta*sigmaOne + (1-beta)*sigmaTwo).determinant() / (pow(sigmaOne.determinant(), beta) * pow(sigmaTwo.determinant(), 1 - beta)));
	return exp(-1.0 * kb);
}

int main()
{
	srand(time(NULL));

	ofstream generalOutput;


	generalOutput.open("./results/PA1-Output.txt");

	Matrix2f sigmaOne, sigmaTwo;
	Vector2f muOne, muTwo;

	float priorOne, priorTwo;

	BayesClassifier classifier;

	vector<Vector2f> sampleOne, sampleTwo;
	
	int misclassifiedOne, misclassifiedTwo;


	//================================================
	// Begin Part One Configuration
	//================================================

	muOne << 1, 1;
	muTwo << 6, 6;

	sigmaOne << 2, 0,
				0, 2;
	sigmaTwo << 2, 0,
				0, 2;

	priorOne = priorTwo = 0.5;
	misclassifiedOne = misclassifiedTwo = 0;

	sampleOne = classifier.generateSamples(muOne, sigmaOne);
	sampleTwo = classifier.generateSamples(muTwo, sigmaTwo);

	//================================================
	// End Part One Configuration
	//================================================

	//================================================
	// Begin Part One Tests
	//================================================

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseOne(sampleOne[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 2)
		{
			misclassifiedOne++;
		}
		if(classifier.classifierCaseOne(sampleTwo[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 1)
		{
			misclassifiedTwo++;
		}
	}

	generalOutput << "================================================\n Part One (A) - (Using the Bayesian Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	// Begin Part B Configuration

	misclassifiedOne = misclassifiedTwo = 0;
	priorOne = 0.2;
	priorTwo = 0.8;

	// End Part B Configuration

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseOne(sampleOne[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0), priorOne, priorTwo) == 2)
		{
			misclassifiedOne++;
		}
		if(classifier.classifierCaseOne(sampleTwo[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0), priorOne, priorTwo) == 1)
		{
			misclassifiedTwo++;
		}
	}

	generalOutput << "================================================\n Part One (B) - (Using the Bayesian Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	writeSamplesToFile("./results/Part-One.txt", sampleOne, sampleTwo);


	//================================================
	// End Part One Tests
	//================================================

	//================================================
	// Begin Part Two Configuration
	//================================================

	muOne << 1, 1;
	muTwo << 6, 6;

	sigmaOne << 2, 0,
				0, 2;
	sigmaTwo << 4, 0,
				0, 8;

	priorOne = priorTwo = 0.5;
	misclassifiedOne = misclassifiedTwo = 0;

	sampleOne = classifier.generateSamples(muOne, sigmaOne);
	sampleTwo = classifier.generateSamples(muTwo, sigmaTwo);

	//================================================
	// End Part Two Configuration
	//================================================

	//================================================
	// Begin Part Two Tests
	//================================================

	vector<Vector2f> sampleMis;

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseThree(sampleOne[i], muOne, muTwo, sigmaOne, sigmaTwo) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(classifier.classifierCaseThree(sampleTwo[i], muOne, muTwo, sigmaOne, sigmaTwo) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}

	generalOutput << "================================================\n Part Two (A) - (Using the Bayesian Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	// Begin Part B Configuration

	misclassifiedOne = misclassifiedTwo = 0;
	priorOne = 0.2;
	priorTwo = 0.8;

	// End Part B Configuration

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseThree(sampleOne[i], muOne, muTwo, sigmaOne, sigmaTwo, priorOne, priorTwo) == 2)
		{
			misclassifiedOne++;
		}
		if(classifier.classifierCaseThree(sampleTwo[i], muOne, muTwo, sigmaOne, sigmaTwo, priorOne, priorTwo) == 1)
		{
			misclassifiedTwo++;
		}
	}

	generalOutput << "================================================\n Part Two (B) - (Using the Bayesian Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	writeSamplesToFile("./results/Part-Two.txt", sampleOne, sampleTwo);
	writeSamplesToFile("./results/Part-Two-Misclassified.txt", sampleMis, sampleMis);

	//================================================
	// End Part Two Tests
	//================================================

	//================================================
	// Begin Part Three Tests
	//================================================

	misclassifiedOne = misclassifiedTwo = 0;

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.minimumDistanceClassifier(sampleOne[i], muOne, muTwo) == 2)
		{
			misclassifiedOne++;
		}
		if(classifier.minimumDistanceClassifier(sampleTwo[i], muOne, muTwo) == 1)
		{
			misclassifiedTwo++;
		}
	}

	generalOutput << "================================================\n Part Three (A) - (Using the Minimum Distance Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	generalOutput.close();

}

void writeSamplesToFile(const char* fileName, vector<Vector2f> sampleOne, vector<Vector2f> sampleTwo)
{
	ofstream partSpecificOutput;

	partSpecificOutput.open(fileName);

	for(unsigned int i = 0; i < sampleOne.size(); i++)
	{
		partSpecificOutput << sampleOne[i](0) << "\t" << sampleOne[i](1) << "\t" << sampleTwo[i](0) << "\t" << sampleTwo[i](1) << endl;
	} 

	partSpecificOutput.close();
}