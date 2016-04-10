#include <iostream>
#include <Eigen/Dense>
#include <dirent.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

using namespace Eigen;
using namespace std;

#include "image.h"

extern float PCA_PERCENTAGE;

bool cmp(pair<string, float> a, pair<string, float> b)
{
    return a.second < b.second;
}

float distanceInFaceSpace(VectorXf originalFace, VectorXf newFace)
{
    return (originalFace - newFace).norm();
}

bool amongNMostSimilarFaces(vector<pair<string, float> > similarFaces, int N, string searchID)
{
    // sort(similarFaces.begin(), similarFaces.end(), cmp);
    for(int i = 0; i < N; i++)
    {
        if(similarFaces[i].first == searchID)
        {
            return true;
        }
    }

    return false;
}

VectorXf projectOntoEigenspace(VectorXf newFace, VectorXf averageFace, MatrixXf eigenfaces)
{
    vector<float> faceCoefficients;
    VectorXf normalizedFace = newFace - averageFace;
    VectorXf projectedFace(averageFace.rows());
    projectedFace.fill(0);
    for(int i = 0; i < eigenfaces.cols(); i++)
    {
        float a = (eigenfaces.col(i).transpose() * normalizedFace)(0,0);
        faceCoefficients.push_back(a);
        projectedFace += (faceCoefficients[i] * eigenfaces.col(i));

    }
    return projectedFace + averageFace;
}

void runClassifier(const char* resultsPath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<pair<string, VectorXf> > trainingFaces, vector<pair<string, VectorXf> > queryFaces)
{
    float eigenValuesSum = eigenvalues.sum();
    float currentEigenTotal = 0;
    int count;
    char fileName[100];

    for(count = 0; currentEigenTotal / eigenValuesSum < PCA_PERCENTAGE && count < eigenvalues.rows(); count++)
    {
        currentEigenTotal += eigenvalues.row(count)(0);
    }

    cout << "Reducing Dimensionality from " << eigenfaces.cols() << " to " << count << "!" << endl;

    MatrixXf reducedEigenFaces(averageFace.rows(), count);

    reducedEigenFaces = eigenfaces.block(0,0,averageFace.rows(),count);
    
    vector<pair<string, VectorXf> > projectedTrainingFaces, projectedQueryFaces;

    for(unsigned int i = 0; i < trainingFaces.size(); i++)
    {
        pair<string, VectorXf> temp(trainingFaces[i].first, projectOntoEigenspace(trainingFaces[i].second, averageFace, reducedEigenFaces));
        projectedTrainingFaces.push_back(temp);
    }
    
    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        pair<string, VectorXf> temp(queryFaces[i].first, projectOntoEigenspace(queryFaces[i].second, averageFace, reducedEigenFaces));
        projectedQueryFaces.push_back(temp);
    }    

    cout << "Reduced!" << endl;

    VectorXf projQueryFace;

    int correct, incorrect; 
    correct = incorrect = 0;
    
    bool querySaved = false;

    ofstream output;

    vector<float> N_Performances(50, 0);

    sprintf(fileName, "%s-%i-NImageNames.txt", resultsPath, (int)(PCA_PERCENTAGE*100));

        output.open(fileName);

    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        cout << "\rQuery Face: " << i << endl;
        projQueryFace = projectedQueryFaces[i].second;
        vector< pair<string, float> > queryPairs; //Pair is training image id (string) and distance (float)

        querySaved = false;

        for(unsigned int t = 0; t < trainingFaces.size(); t++)
        {
            pair<string, float> newPair(trainingFaces[t].first, distanceInFaceSpace(projQueryFace, trainingFaces[t].second));
            queryPairs.push_back(newPair);
        }

        sort(queryPairs.begin(), queryPairs.end(), cmp);

        

        for(int n = 0; n < 50; n++)
        {
            

            if(amongNMostSimilarFaces(queryPairs, n+1, projectedQueryFaces[i].first))
            {
                N_Performances[n] += 1;
                if(correct < 3 && !querySaved && n == 1)
                {
                    output << "Cor Query Img " << correct << " ID: " << queryFaces[i].first;
                    output << " Cor Train Img " << correct << " ID: " << queryPairs[0].first;
                    output << endl << endl;
                    correct++;
                    querySaved = true;
                }
            }
            else
            {
                if(incorrect < 3 && !querySaved && n == 1)
                {
                    output << "Inc Query Img " << incorrect << " ID: " << queryFaces[i].first;
                    output << " Inc Train Img " << incorrect << " ID: " << queryPairs[0].first;
                    output << endl << endl;
                    incorrect++;
                    querySaved = true;
                }
            }
        }

        // if(amongNMostSimilarFaces(queryPairs, 50, projectedQueryFaces[i].first))
        // {
        //     cout << "Among N most similarFaces : Yes";
        //     correct++;
        // }
        // else
        // {
        //     cout << "Among N most similarFaces : No";
        //     incorrect++;
        // }

        // cout << "\t Percentage Correct So Far = " << ((float)correct / (float)(correct + incorrect)) << endl;
    }

    output.close();

    sprintf(fileName, "%s-%i.txt", resultsPath, (int)(PCA_PERCENTAGE*100));

    output.open(fileName);

    for(int n = 0; n < 50; n++)
    {
        output << n+1 << "\t" << (N_Performances[n] / (float)queryFaces.size()) << endl;
    }

    output.close();


}

    
