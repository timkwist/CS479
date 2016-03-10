#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"

#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

void writeSamplesToFile(const char* fileName, vector<Vector2f> sampleOne, vector<Vector2f> sampleTwo);

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
	vector<Vector2f> sampleMis;
	
	int misclassifiedOne, misclassifiedTwo;

	pair<float, float> chernoffBound; // <index, value>




	//================================================
	// Begin Part One Configuration
	//================================================

	muOne << 1.0, 1.0;
	muTwo << 6.0, 6.0;

	sigmaOne << 2.0, 0.0,
				0.0, 2.0;
	sigmaTwo << 2.0, 0.0,
				0.0, 2.0;

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
			sampleMis.push_back(sampleOne[i]);
		}
		if(classifier.classifierCaseOne(sampleTwo[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}

	writeSamplesToFile("./results/Part-OneA-Misclassified.txt", sampleMis, sampleMis);

	generalOutput << "================================================\n Part One (A) - (Using the Bayesian Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	// Begin Part B Configuration

	misclassifiedOne = misclassifiedTwo = 0;
	sampleMis.clear();
	priorOne = 0.2;
	priorTwo = 0.8;

	// End Part B Configuration

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseOne(sampleOne[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0), priorOne, priorTwo) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(classifier.classifierCaseOne(sampleTwo[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0), priorOne, priorTwo) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}

	chernoffBound = classifier.findChernoffBound(muOne, muTwo, sigmaOne, sigmaTwo);

	generalOutput << "================================================\n Part One (B) - (Using the Bayesian Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	generalOutput << "================================================\n Part One - Error Bounds \n================================================" << endl;
	generalOutput << "With beta = " << chernoffBound.first << " , Chernoff Bound = " << chernoffBound.second << endl;
	generalOutput << "With beta = 0.5, Bhattacharyya Bound = " << classifier.findBhattacharyyaBound(muOne, muTwo, sigmaOne, sigmaTwo) << endl;

	writeSamplesToFile("./results/Part-One.txt", sampleOne, sampleTwo);
	writeSamplesToFile("./results/Part-OneB-Misclassified.txt", sampleMis, sampleMis);
	


	//================================================
	// End Part One Tests
	//================================================

	//================================================
	// Begin Part Two Configuration
	//================================================

	muOne << 1.0, 1.0;
	muTwo << 6.0, 6.0;

	sigmaOne << 2.0, 0.0,
				0.0, 2.0;
	sigmaTwo << 4.0, 0.0,
				0.0, 8.0;

	priorOne = priorTwo = 0.5;
	misclassifiedOne = misclassifiedTwo = 0;

	sampleOne = classifier.generateSamples(muOne, sigmaOne);
	sampleTwo = classifier.generateSamples(muTwo, sigmaTwo);

	sampleMis.clear();

	//================================================
	// End Part Two Configuration
	//================================================

	//================================================
	// Begin Part Two Tests
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

	generalOutput << "================================================\n Part Two (A) - (Using the Bayesian Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	writeSamplesToFile("./results/Part-TwoA-Misclassified.txt", sampleMis, sampleMis);

	// Begin Part B Configuration

	misclassifiedOne = misclassifiedTwo = 0;
	priorOne = 0.2;
	priorTwo = 0.8;

	sampleMis.clear();

	// End Part B Configuration

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.classifierCaseThree(sampleOne[i], muOne, muTwo, sigmaOne, sigmaTwo, priorOne, priorTwo) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(classifier.classifierCaseThree(sampleTwo[i], muOne, muTwo, sigmaOne, sigmaTwo, priorOne, priorTwo) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}

	chernoffBound = classifier.findChernoffBound(muOne, muTwo, sigmaOne, sigmaTwo);

	generalOutput << "================================================\n Part Two (B) - (Using the Bayesian Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	generalOutput << "================================================\n Part Two - Error Bounds \n================================================" << endl;
	generalOutput << "With beta = " << chernoffBound.first << " , Chernoff Bound = " << chernoffBound.second << endl;
	generalOutput << "With beta = 0.5, Bhattacharyya Bound = " << classifier.findBhattacharyyaBound(muOne, muTwo, sigmaOne, sigmaTwo) << endl;

	writeSamplesToFile("./results/Part-Two.txt", sampleOne, sampleTwo);
	writeSamplesToFile("./results/Part-TwoB-Misclassified.txt", sampleMis, sampleMis);

	//================================================
	// End Part Two Tests
	//================================================

	//================================================
	// Begin Part Three Tests
	//================================================

	misclassifiedOne = misclassifiedTwo = 0;
	sampleMis.clear();

	for(int i = 0; i < 10000; i++)
	{
		if(classifier.minimumDistanceClassifier(sampleOne[i], muOne, muTwo) == 2)
		{
			misclassifiedOne++;
			sampleMis.push_back(sampleOne[i]);
		}
		if(classifier.minimumDistanceClassifier(sampleTwo[i], muOne, muTwo) == 1)
		{
			misclassifiedTwo++;
			sampleMis.push_back(sampleTwo[i]);
		}
	}

	generalOutput << "================================================\n Part Three (A) - (Using the Minimum Distance Classifier) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << endl;
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << endl;
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << endl;

	writeSamplesToFile("./results/Part-Three-Misclassified.txt", sampleMis, sampleMis);

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