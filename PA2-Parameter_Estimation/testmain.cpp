#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"
#include "SampleGenerator.h"
#include "MLE.h"
#include <cstdlib>

using namespace std;
using namespace Eigen;

#include "image.h"

int readImageHeader(char[], int&, int&, int&, bool&);
int readImage(char[], ImageType&);
int writeImage(char[], ImageType&);

void getMLEParameters(ImageType& trainingImage, ImageType& refImage, bool useRGB, Vector2f &estSkinMu, Matrix2f &estSkinSigma, Vector2f &estNonSkinMu, Matrix2f &estNonSkinSigma);
void runThresholdTest(ImageType& testImage, ImageType& refImage, bool useRGB, float thresMin, float thresMax, Vector2f &estMu, Matrix2f &estSigma, const char fileOutput[]);
void runTwoClassTest(ImageType& testImage, ImageType& refImage, bool useRGB, Vector2f &estSkinMu, Matrix2f &estSkinSigma, Vector2f &estNonSkinMu, Matrix2f &estNonSkinSigma, const char fileOutput[]);

int M, N, Q;

int main(int argc, char *argv[])
{	
	bool type;
	RGB val;
	vector<Vector2f> sampleData;

 	// read training images' headers
	readImageHeader("ref1.ppm", N, M, Q, type);
	readImageHeader("train1.ppm", N, M, Q, type);

 	// allocate memory for the image array
	ImageType refernceImage(N, M, Q);
	ImageType testImage(N, M, Q);

 	// read image
	readImage("train1.ppm", testImage);
	readImage("ref1.ppm", refernceImage);

 	// Create the model from the training data

	Vector2f estSkinMuRGB, estSkinMuYCC, estNonSkinMuRGB, estNonSkinMuYCC;
	Matrix2f estSkinSigmaRGB, estSkinSigmaYCC, estNonSkinSigmaRGB, estNonSkinSigmaYCC;
	
	getMLEParameters(testImage, refernceImage, true, estSkinMuRGB, estSkinSigmaRGB, estNonSkinMuRGB, estNonSkinSigmaRGB);
	getMLEParameters(testImage, refernceImage, false, estSkinMuYCC, estSkinSigmaYCC, estNonSkinMuYCC, estNonSkinSigmaYCC);

	readImage("train3.ppm", testImage);
	readImage("ref3.ppm", refernceImage);

	cout << "Running tests for train3.ppm (RGB) ... " << endl;
	runThresholdTest(testImage, refernceImage, true, -1, 0, estSkinMuRGB, estSkinSigmaRGB, "Train3-RGB-ROC-Data.txt");

	cout << "Running tests for train3.ppm (YCbCr) ... " << endl;
	runThresholdTest(testImage, refernceImage, false, -1, 0, estSkinMuYCC, estSkinSigmaYCC, "Train3-YCC-ROC-Data.txt");

	readImage("train6.ppm", testImage);
	readImage("ref6.ppm", refernceImage);

	cout << "Running tests for train6.ppm (RGB) ... " << endl;
	runThresholdTest(testImage, refernceImage, true, -1, 0, estSkinMuRGB, estSkinSigmaRGB, "Train6-RGB-ROC-Data.txt");

	cout << "Running tests for train6.ppm (YCbCr) ... " << endl;
	runThresholdTest(testImage, refernceImage, false, -1, 0, estSkinMuYCC, estSkinSigmaYCC, "Train6-YCC-ROC-Data.txt");


	return (1);
}

void getMLEParameters(ImageType& trainingImage, ImageType& refImage, bool useRGB, Vector2f &estSkinMu, Matrix2f &estSkinSigma, Vector2f &estNonSkinMu, Matrix2f &estNonSkinSigma)
{
	vector<Vector2f> sampleSkinData, sampleNonSkinData;

	RGB val;

 	float total, x1, x2;

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<M; j++) 
		{
			trainingImage.getPixelVal(i, j, val);
			if(useRGB)
			{
				total = val.r + val.g + val.b;
				x1 = (float)val.r / total;	//New Red value
				x2 = (float)val.g / total;  //New Green Value
			}
			else
			{
				x1 = -0.169 * (float)val.r - 0.332 * (float)val.g + 0.5 * (float)val.b; //New Cb value
				x2 = 0.5 * (float)val.r - 0.419 * (float)val.g - 0.081 * (float)val.b;  //New Cr value
			}
			refImage.getPixelVal(i, j, val);
			if(val.r != 0 && val.g != 0 && val.b != 0)
			{
				sampleSkinData.push_back(Vector2f(x1, x2));
			}
			else
			{
				sampleNonSkinData.push_back(Vector2f(x1, x2));
			}
		}
	}

	estSkinMu = MLE::calculateSampleMean(sampleSkinData);
	estSkinSigma = MLE::calculateSampleCovariance(sampleSkinData, estSkinMu);
	estNonSkinMu = MLE::calculateSampleMean(sampleNonSkinData);
	estNonSkinSigma = MLE::calculateSampleCovariance(sampleNonSkinData, estNonSkinMu);
}

void runThresholdTest(ImageType& testImage, ImageType& refImage, bool useRGB, float thresMin, float thresMax, Vector2f &estMu, Matrix2f &estSigma, const char fileOutput[])
{
	vector<float> falseNegative, falsePositive;
	float n, p, fn, fp, x1, x2, total;
	RGB val;

	for(float threshold = thresMin; threshold <= thresMax+0.02; threshold+=.05)
	{
		n = p = fn = fp = 0;
		for(int i=0; i<N; i++)
		{
			for(int j=0; j<M; j++) 
			{
				testImage.getPixelVal(i, j, val);

				if(useRGB)
				{
					total = val.r + val.g + val.b;
					x1 = (float)val.r / total;	//New Red value
					x2 = (float)val.g / total;  //New Green Value
				}
				else
				{
					x1 = -0.169 * (float)val.r - 0.332 * (float)val.g + 0.5 * (float)val.b; //New Cb value
					x2 = 0.5 * (float)val.r - 0.419 * (float)val.g - 0.081 * (float)val.b;  //New Cr value
				}

				bool classifiedAsSkin = BayesClassifier::thresholdCaseThree(Vector2f(x1, x2), estMu, estSigma, threshold);
				 
				refImage.getPixelVal(i, j, val);

				bool isSkin = (val.r != 0 && val.g != 0 && val.b != 0);

				if(classifiedAsSkin)
					p++;
				else
					n++;
				
				if(isSkin && !classifiedAsSkin)
				{
					fn++;
				}
				else if(!isSkin && classifiedAsSkin)
				{
					fp++;
				}
			}
		}

		// cout << "Threshold: " << threshold << ": " << endl;
		// cout << "\tFalse Negative Rate: \t" << fn / n << endl;
		// cout << "\tFalse Positive Rate: \t" << fp / p << endl << endl;

		falseNegative.push_back(fn / n);
		falsePositive.push_back(fp / p);
	}
	
	ofstream generalOutput;

	generalOutput.open(fileOutput);

	generalOutput << "Threshold\tFalseNegative\tFalsePositive" << endl;

	for(float threshold = thresMin, i = 0; threshold <= thresMax+0.02; threshold+=.05, i++)
	{
		generalOutput << threshold << "\t" << falseNegative[i] << "\t" << falsePositive[i] << endl;
	}

	generalOutput.close();
}

void runTwoClassTest(ImageType& testImage, ImageType& refImage, bool useRGB, Vector2f &estSkinMu, Matrix2f &estSkinSigma, Vector2f &estNonSkinMu, Matrix2f &estNonSkinSigma, const char fileOutput[])
{
	float falseNegative, falsePositive;
	float n, p, fn, fp, x1, x2, total;
	RGB val;

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<M; j++) 
		{
			testImage.getPixelVal(i, j, val);
			if(useRGB)
			{
				total = val.r + val.g + val.b;
				x1 = (float)val.r / total;	//New Red value
				x2 = (float)val.g / total;  //New Green Value
			}
			else
			{
				x1 = -0.169 * (float)val.r - 0.332 * (float)val.g + 0.5 * (float)val.b; //New Cb value
				x2 = 0.5 * (float)val.r - 0.419 * (float)val.g - 0.081 * (float)val.b;  //New Cr value
			}

			bool classifiedAsSkin = BayesClassifier::classifierCaseThree(Vector2f(x1, x2), estSkinMu, estNonSkinMu, estSkinSigma, estNonSkinSigma);
			 
			refImage.getPixelVal(i, j, val);

			bool isSkin = (val.r != 0 && val.g != 0 && val.b != 0);

			if(classifiedAsSkin)
				p++;
			else
				n++;
				
			if(isSkin && !classifiedAsSkin)
			{
				fn++;
			}
			else if(!isSkin && classifiedAsSkin)
			{
				fp++;
			}
		}
	}
	
	falseNegative = fn / n;
	falsePositive = fp / p;
	
	ofstream generalOutput;

	generalOutput.open(fileOutput);

	generalOutput << "Two-Class (Skin vs Non-Skin) Results:" << endl;

	generalOutput << "False Negative: " << falseNegative << endl;
	generalOutput << "False Positive: " << falsePositive << endl;

	generalOutput.close();
}
