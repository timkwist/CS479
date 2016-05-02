#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"
#include "MLE.h"
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>
#include <sstream>


using namespace Eigen;
using namespace std;


void readInFaces(vector<VectorXf> &maleFaces, vector<VectorXf> &femaleFaces, string examplesFile, string targetFile);

int main()
{

    // Read in Training face vectors

    // cout << "Reading in training data!" << endl;

    vector<VectorXf> maleTrainingFaces;
    vector<VectorXf> femaleTrainingFaces;

    readInFaces(maleTrainingFaces, femaleTrainingFaces, "BayesData/48_60/new-trPCA_03.txt", "BayesData/48_60/TtrPCA_03.txt");
    //readInFaces(maleTrainingFaces, femaleTrainingFaces, "example.txt", "targets.txt");
    
    // cout <<

    // Calculate covariance and mean

    // cout << "Calculating Means and Covariance Matricies" << endl;

    VectorXf maleMean       = MLE::calculateSampleMean(maleTrainingFaces);
    VectorXf femaleMean     = MLE::calculateSampleMean(femaleTrainingFaces);

    MatrixXf maleCovar      = MLE::calculateSampleCovariance(maleTrainingFaces, maleMean);
    MatrixXf femaleCovar    = MLE::calculateSampleCovariance(femaleTrainingFaces, femaleMean);

    cout << maleCovar << endl;

    // Run Classifiers on test and validation data

    // cout << "Reading in test data!" << endl;

    // vector<VectorXf> maleTestFaces;
    // vector<VectorXf> femaleTestFaces;

    // readInFaces(maleTestFaces, femaleTestFaces, "BayesData/48_60/testPCA_03.txt", "BayesData/48_60/TtestPCA_03.txt");

    // cout << "Running classifier on test data!" << endl;

    // int correct = 0;

    // for(unsigned int i = 0; i < maleTestFaces.size(); i++)
    // {
    //     cout << i << endl;
    //     if(BayesClassifier::classifierCaseThree(maleTestFaces[i], maleMean, femaleMean, maleCovar, femaleCovar) == 1)
    //     {
    //         correct++;
    //     }
    // }

    // //Minus 8 here because for some reason femaleFaces seem to be corrupt or I might have copied the data wrong..

    // for(unsigned int i = 0; i < femaleTestFaces.size(); i++)
    // {
    //     cout << i << endl;
    //     if(BayesClassifier::classifierCaseThree(femaleTestFaces[i], maleMean, femaleMean, maleCovar, femaleCovar) == 2)
    //     {
    //         correct++;
    //     }
    // }
    // cout << endl << "RESULTS: " << endl;

    // cout << "\tTotal tests: \t" << (maleTestFaces.size() + femaleTestFaces.size()) << endl;
    // cout << "\tCorrect:     \t" << correct << endl;
    // cout << "\tPercentage:  \t" << ((float)correct / (float)(maleTestFaces.size() + femaleTestFaces.size())) << endl;

    // // Print results 

	return 0;
}

void readInFaces(vector<VectorXf> &maleFaces, vector<VectorXf> &femaleFaces, string examplesFilePath, string targetFilePath)
{
    maleFaces.clear();
    femaleFaces.clear();

    ifstream faceFile(examplesFilePath);
    ifstream targetFile(targetFilePath);
    
    string line;
    
    getline(targetFile, line);

    istringstream targetStream(line);

    while (getline(faceFile, line))
    {
        istringstream iss(line);

        vector<float> values;

        float val = 0;

        while(true)
        {
            if (! (iss >> val) ) { break; }
            values.push_back(val);
        }

        VectorXf newFace(values.size());

        for(unsigned int i = 0; i < values.size(); i++)
        {
            newFace.row(i) << values[i];
        }

        int bayesClass = 0;

        if (! (targetStream >> bayesClass) ) { cout << "Error reading targets!" << endl; break; }

        if(bayesClass == 1)
            maleFaces.push_back(newFace);
        else
            femaleFaces.push_back(newFace);
            
    }
}