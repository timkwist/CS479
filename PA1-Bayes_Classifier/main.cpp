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

	generalOutput << "================================================\n Part One (A) \n================================================" << endl;
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

	generalOutput << "================================================\n Part One (B) \n================================================" << endl;
	generalOutput << "Samples from one misclassified: " << misclassifiedOne << "\n";
	generalOutput << "Samples from two misclassified: " << misclassifiedTwo << "\n";
	generalOutput << "Total misclassified: " << misclassifiedOne + misclassifiedTwo << "\n";

	writeSamplesToFile("./results/Part-One.txt", sampleOne, sampleTwo);


	//================================================
	// End Part One Tests
	//================================================


	// ofstream output;
	// output.open("output.txt");

	// for(int i = 0; i < 10000; i++)
	// {
	// 	output << sampleOne[i](0) << "\t" << sampleOne[i](1) << "\t" << sampleTwo[i](0) << "\t" << sampleTwo[i](1) << endl;
	// }

	// output.close();

	generalOutput.close();
	// partSpecificOutput.close();

}

void writeSamplesToFile(const char* fileName, vector<Vector2f> sampleOne, vector<Vector2f> sampleTwo)
{
	ofstream partSpecificOutput;

	partSpecificOutput.open(fileName);

	for(int i = 0; i < 10000; i++)
	{
		partSpecificOutput << sampleOne[i](0) << "\t" << sampleOne[i](1) << "\t" << sampleTwo[i](0) << "\t" << sampleTwo[i](1) << endl;
	} 

	partSpecificOutput.close();
}