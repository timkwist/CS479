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
		if(BayesClassifier::classifierCaseOne(sampleOne[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(BayesClassifier::classifierCaseOne(sampleTwo[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 1)
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
		if(BayesClassifier::classifierCaseOne(sampleOne[i], estMuOne, estMuTwo, estSigmaOne(0,0), estSigmaTwo(0,0)) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(BayesClassifier::classifierCaseOne(sampleTwo[i], estMuOne, estMuTwo, estSigmaOne(0,0), estSigmaTwo(0,0)) == 1)
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
	// Begin Part 1B Tests (Esimated Covaraince and Mean & Small Sample)
	//================================================

	vector<Vector2f> smallSampleOne, smallSampleTwo;

	for(int i = 0; i < 1000; i++)
	{
		int idxOne = randIndex(sampleOne.size());
		int idxTwo = randIndex(sampleTwo.size());

		smallSampleOne.push_back(sampleOne[idxOne]); sampleOne.erase(sampleOne.begin() + idxOne);
		smallSampleTwo.push_back(sampleTwo[idxTwo]); sampleTwo.erase(sampleTwo.begin() + idxTwo);
	}
	
	estMuOne = MLE::calculateSampleMean(smallSampleOne);
	estMuTwo = MLE::calculateSampleMean(smallSampleTwo);

	estSigmaOne = MLE::calculateSampleCovariance(smallSampleOne, estMuOne);
	estSigmaTwo = MLE::calculateSampleCovariance(smallSampleTwo, estMuTwo);

	sampleMis.clear();
	misclassifiedOne = misclassifiedTwo = 0;

	for(int i = 0; i < 10000; i++)
	{
		if(BayesClassifier::classifierCaseOne(sampleOne[i], estMuOne, estMuTwo, estSigmaOne(0,0), estSigmaTwo(0,0)) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(BayesClassifier::classifierCaseOne(sampleTwo[i], estMuOne, estMuTwo, estSigmaOne(0,0), estSigmaTwo(0,0)) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}


	writeSamplesToFile("./results/Part1B-Estimated-Misclassified.txt", sampleMis, sampleMis);

	generalOutput << "================================================\n Part 1B - (Esimated Parameters & Small Sample) \n================================================" << endl;
	generalOutput << "Estimated Sample Mean: muOne=[" << estMuOne(0) << ", " << estMuOne(1) << "]" <<
					 " muTwo=[" << estMuTwo(0) << ", " << estMuTwo(1) << "]" << endl;
	generalOutput << "Estimated Sample Covaraince: \nsigmaOne" << endl;
	generalOutput << estSigmaOne << endl << "sigmaTwo" << endl << estSigmaTwo << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	//================================================
	// End Part 1B Tests (Esimated Covaraince and Mean & Small Sample)
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
		if(BayesClassifier::classifierCaseThree(sampleOne[i], muOne, muTwo, sigmaOne, sigmaTwo) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(BayesClassifier::classifierCaseThree(sampleTwo[i], muOne, muTwo, sigmaOne, sigmaTwo) == 1)
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
		if(BayesClassifier::classifierCaseThree(sampleOne[i], estMuOne, estMuTwo, estSigmaOne, estSigmaTwo) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(BayesClassifier::classifierCaseThree(sampleTwo[i], estMuOne, estMuTwo, estSigmaOne, estSigmaTwo) == 1)
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

	//================================================
	// Begin Part 2B Tests (Esimated Covaraince and Mean & Small Sample)
	//================================================

	smallSampleOne.clear();
	smallSampleTwo.clear();

	for(int i = 0; i < 1000; i++)
	{
		int idxOne = randIndex(sampleOne.size());
		int idxTwo = randIndex(sampleTwo.size());

		smallSampleOne.push_back(sampleOne[idxOne]); sampleOne.erase(sampleOne.begin() + idxOne);
		smallSampleTwo.push_back(sampleTwo[idxTwo]); sampleTwo.erase(sampleTwo.begin() + idxTwo);
	}
	
	estMuOne = MLE::calculateSampleMean(smallSampleOne);
	estMuTwo = MLE::calculateSampleMean(smallSampleTwo);

	estSigmaOne = MLE::calculateSampleCovariance(smallSampleOne, estMuOne);
	estSigmaTwo = MLE::calculateSampleCovariance(smallSampleTwo, estMuTwo);

	sampleMis.clear();
	misclassifiedOne = misclassifiedTwo = 0;

	for(int i = 0; i < 10000; i++)
	{
		if(BayesClassifier::classifierCaseOne(sampleOne[i], estMuOne, estMuTwo, estSigmaOne(0,0), estSigmaTwo(0,0)) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(BayesClassifier::classifierCaseOne(sampleTwo[i], estMuOne, estMuTwo, estSigmaOne(0,0), estSigmaTwo(0,0)) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}


	writeSamplesToFile("./results/Part2B-Estimated-Misclassified.txt", sampleMis, sampleMis);

	generalOutput << "================================================\n Part 2B - (Esimated Parameters & Small Sample) \n================================================" << endl;
	generalOutput << "Estimated Sample Mean: muOne=[" << estMuOne(0) << ", " << estMuOne(1) << "]" <<
					 " muTwo=[" << estMuTwo(0) << ", " << estMuTwo(1) << "]" << endl;
	generalOutput << "Estimated Sample Covaraince: \nsigmaOne" << endl;
	generalOutput << estSigmaOne << endl << "sigmaTwo" << endl << estSigmaTwo << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	//================================================
	// End Part 2B Tests (Esimated Covaraince and Mean & Small Sample)
	//================================================

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