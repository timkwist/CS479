#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"
#include "SampleGenerator.h"
#include "MLE.h"
#include <cstdlib>

#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

void writeSamplesToFile(const char* fileName, vector<Vector2f> sampleOne, vector<Vector2f> sampleTwo);

int randIndex(int size);

int main()
{
	srand(time(NULL));

	ofstream generalOutput;

	generalOutput.open("./results/PA1-Output.txt");

	Matrix2f sigmaOne, sigmaTwo;
	Vector2f muOne, muTwo;

	BayesClassifier classifier;
	SampleGenerator generator;

	vector<Vector2f> sampleOne, sampleTwo;
	vector<Vector2f> sampleMis;

	Matrix2f estSigmaOne, estSigmaTwo;
	Vector2f estMuOne, estMuTwo;
	
	int misclassifiedOne, misclassifiedTwo;

	//================================================
	// Begin Part 1A Configuration 
	//================================================

	muOne << 1.0, 1.0;
	muTwo << 6.0, 6.0;

	sigmaOne << 2.0, 0.0,
				0.0, 2.0;
	sigmaTwo << 2.0, 0.0,
				0.0, 2.0;

	misclassifiedOne = misclassifiedTwo = 0;

	sampleOne = generator.generateSamples(muOne, sigmaOne);
	sampleTwo = generator.generateSamples(muTwo, sigmaTwo);

	//================================================
	// End Part 1A Configuration 
	//================================================

	//================================================
	// Begin Part 1A Tests (Known Covaraince and Mean)
	//================================================

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseOne(sampleOne[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(classifier.classifierCaseOne(sampleTwo[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}

	writeSamplesToFile("./results/Part1A-Known-Misclassified.txt", sampleMis, sampleMis);

	generalOutput << "================================================\n Part 1A - (Known Parameters) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	//================================================
	// End Part 1A Tests (Known Covaraince and Mean)
	//================================================

	//================================================
	// Begin Part 1A Tests (Esimated Covaraince and Mean)
	//================================================

	estMuOne = MLE::calculateSampleMean(sampleOne);
	estMuTwo = MLE::calculateSampleMean(sampleTwo);

	estSigmaOne = MLE::calculateSampleCovariance(sampleOne, estMuOne);
	estSigmaTwo = MLE::calculateSampleCovariance(sampleTwo, estMuTwo);

	sampleMis.clear();
	misclassifiedOne = misclassifiedTwo = 0;

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseOne(sampleOne[i], estMuOne, estMuTwo, estSigmaOne(0,0), estSigmaTwo(0,0)) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(classifier.classifierCaseOne(sampleTwo[i], estMuOne, estMuTwo, estSigmaOne(0,0), estSigmaTwo(0,0)) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}


	writeSamplesToFile("./results/Part1A-Estimated-Misclassified.txt", sampleMis, sampleMis);

	generalOutput << "================================================\n Part 1A - (Esimated Parameters) \n================================================" << endl;
	generalOutput << "Estimated Sample Mean: muOne=[" << estMuOne(0) << ", " << estMuOne(1) << "]" <<
					 " muTwo=[" << estMuTwo(0) << ", " << estMuTwo(1) << "]" << endl;
	generalOutput << "Estimated Sample Covaraince: \nsigmaOne" << endl;
	generalOutput << estSigmaOne << endl << "sigmaTwo" << endl << estSigmaTwo << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	writeSamplesToFile("./results/Part1A-Samples.txt", sampleOne, sampleTwo);

	//================================================
	// End Part 1A Tests (Esimated Covaraince and Mean)
	//================================================


	











	//================================================
	// Begin Part 2A Configuration
	//================================================

	muOne << 1.0, 1.0;
	muTwo << 6.0, 6.0;

	sigmaOne << 2.0, 0.0,
				0.0, 2.0;
	sigmaTwo << 4.0, 0.0,
				0.0, 8.0;

	misclassifiedOne = misclassifiedTwo = 0;

	sampleOne = generator.generateSamples(muOne, sigmaOne);
	sampleTwo = generator.generateSamples(muTwo, sigmaTwo);

	sampleMis.clear();

	//================================================
	// Begin Part 2A Tests (Known Covaraince and Mean)
	//================================================

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

	writeSamplesToFile("./results/Part2A-Known-Misclassified.txt", sampleMis, sampleMis);

	generalOutput << "================================================\n Part 2A - (Known Parameters) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	//================================================
	// End Part 2A Tests (Known Covaraince and Mean)
	//================================================

	//================================================
	// Begin Part 2A Tests (Esimated Covaraince and Mean)
	//================================================

	estMuOne = MLE::calculateSampleMean(sampleOne);
	estMuTwo = MLE::calculateSampleMean(sampleTwo);

	estSigmaOne = MLE::calculateSampleCovariance(sampleOne, estMuOne);
	estSigmaTwo = MLE::calculateSampleCovariance(sampleTwo, estMuTwo);

	sampleMis.clear();
	misclassifiedOne = misclassifiedTwo = 0;

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseThree(sampleOne[i], estMuOne, estMuTwo, estSigmaOne, estSigmaTwo) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(classifier.classifierCaseThree(sampleTwo[i], estMuOne, estMuTwo, estSigmaOne, estSigmaTwo) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}

	writeSamplesToFile("./results/Part2A-Estimated-Misclassified.txt", sampleMis, sampleMis);

	generalOutput << "================================================\n Part 2A - (Esimated Parameters) \n================================================" << endl;
	generalOutput << "Estimated Sample Mean: muOne=[" << estMuOne(0) << ", " << estMuOne(1) << "]" <<
					 " muTwo=[" << estMuTwo(0) << ", " << estMuTwo(1) << "]" << endl;
	generalOutput << "Estimated Sample Covaraince: \nsigmaOne" << endl;
	generalOutput << estSigmaOne << endl << "sigmaTwo" << endl << estSigmaTwo << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	writeSamplesToFile("./results/Part2A-Samples.txt", sampleOne, sampleTwo);

	generalOutput.close();

}

int randIndex(int size)
{
	return (rand() / (float)RAND_MAX) * size;
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