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

/**
 * Comparison operator for pair of string / float
 * Returns true if the float in a is less than float in b
 * Returns false otherwise
 * 
 * @param  a Pair consisting of string and float
 * @param  b Pair consisting of string and float
 * @return   True if the float in a is less than the float in b; false otherwise
 */
bool cmp(pair<string, float> a, pair<string, float> b)
{
    return a.second < b.second;
}

/**
 * Returns the distance in face space,
 * defined as the norm of the original face subtracted by
 * the new face.
 * 
 * @param  originalFace Face before projection (or face already in face space)
 * @param  newFace      Unclassified face after it has been projected
 * @return              The distance in face space
 */
float distanceInFaceSpace(VectorXf originalFace, VectorXf newFace)
{
    return (originalFace - newFace).norm();
}

/**
 * Searches through the top N most similar faces (which are the N smallest distances in face space)
 * Returns true if one of those faces has the same ID as the search ID
 * 
 * @param  similarFaces Vector of faces, with the string defining the face's ID and the float defining
 *                      the distance in face space
 * @param  N            Number of faces to search through
 * @param  searchID     Desired ID that is being searched for
 * @return              True if the desired ID matches one of the N most similar faces; false otherwise
 */
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

/**
 * Calculates the projection of the new face onto the previously defined face space
 * 
 * @param  newFace     The unclassified face that will be projected
 * @param  averageFace The average face of the face space
 * @param  eigenfaces  The top K eigenfaces(eigenvectors) that make up / represent the face space
 * @return             Vector of size equivalent to newFace and averageFace that represents the
 *                            new face projected onto the eigenspace defined by the average face
 *                            and eigenfaces
 */
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

/**
 * Runs the default classifer that cycles from 1 <= N <= 50 and prints its results to a series of file
 * in the specified directory. It prints out the correctly / incorrectly matched faces along with the
 * data used to create the CMC curve.
 * 
 * @param  resultsPath      The directory to save the result files.
 * @param  averageFace      The average face of the face space
 * @param  eigenfaces       The eigenfaces(eigenvectors) that make up / represent the face space
 * @param  eigenvalues      The sorted eigenvalues that represent the best eigenfaces to use for
 *                            classification and information preservation
 *                            
 * @param  trainingFaces    A vector of faces that were used to create the eigefaces and eigenvalues
 * @param  queryFaces       A vector of faces that will be used to query the training data
 * 
 */
void runClassifier(const char* resultsPath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<pair<string, VectorXf> > trainingFaces, vector<pair<string, VectorXf> > queryFaces)
{
    // Perform PCA dimensionality reduction

    float eigenValuesSum = eigenvalues.sum();
    float currentEigenTotal = 0;
    int count;
    char fileName[100];

    //Find the number of vectors to preserve PCA_PERCENTAGE of the information
    for(count = 0; currentEigenTotal / eigenValuesSum < PCA_PERCENTAGE && count < eigenvalues.rows(); count++)
    {
        currentEigenTotal += eigenvalues.row(count)(0);
    }

    cout << "Reducing Dimensionality from " << eigenfaces.cols() << " to " << count << "!" << endl;

    MatrixXf reducedEigenFaces(averageFace.rows(), count);

    reducedEigenFaces = eigenfaces.block(0,0,averageFace.rows(),count);
        
    //Project the faces onto the reduced eigenfaces

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

    // Iterate through each query face and see if it can be classified correctly

    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        projQueryFace = projectedQueryFaces[i].second;

        vector< pair<string, float> > queryPairs; //Pair is training image id (string) and distance (float)

        //We saved this query as a correct / incorrect match
        querySaved = false;

        //Find the distances from this query face to every training face
        for(unsigned int t = 0; t < trainingFaces.size(); t++)
        {
            pair<string, float> newPair(trainingFaces[t].first, distanceInFaceSpace(projQueryFace, projectedTrainingFaces[t].second));
            queryPairs.push_back(newPair);
        }

        // Sort the distances from least to greatest
        sort(queryPairs.begin(), queryPairs.end(), cmp);

        // Iterate from n = 1 to 50
        for(int n = 0; n < 50; n++)
        {

            if(amongNMostSimilarFaces(queryPairs, n+1, projectedQueryFaces[i].first))
            {
                N_Performances[n] += 1;
                //Only save a correct match if N = 1 (0 in this case since we start at 0)
                if(correct < 3 && !querySaved && n == 0)
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
                //Only save an incorrect match if N = 1 (0 in this case since we start at 0)
                if(incorrect < 3 && !querySaved && n == 0)
                {
                    output << "Inc Query Img " << incorrect << " ID: " << queryFaces[i].first;
                    output << " Inc Train Img " << incorrect << " ID: " << queryPairs[0].first;
                    output << endl << endl;
                    incorrect++;
                    querySaved = true;
                }
            }
        }
    }

    output.close();

    sprintf(fileName, "%s-%i.txt", resultsPath, (int)(PCA_PERCENTAGE*100));

    output.open(fileName);
    
    //Print out the data for the CMC curve
    for(int n = 0; n < 50; n++)
    {
        output << n+1 << "\t" << (N_Performances[n] / (float)queryFaces.size()) << endl;
    }

    output.close();

}

/**
 * Runs the threshold classifer that varies a threshold to determine if a face is allowed or not and 
 * prints its results to a series of file in the specified directory. It prints out the data to create
 * the ROC curve.
 * 
 * @param  resultsPath      The directory to save the result files.
 * @param  averageFace      The average face of the face space
 * @param  eigenfaces       The eigenfaces(eigenvectors) that make up / represent the face space
 * @param  eigenvalues      The sorted eigenvalues that represent the best eigenfaces to use for
 *                            classification and information preservation
 *                            
 * @param  trainingFaces    A vector of faces that were used to create the eigefaces and eigenvalues
 * @param  queryFaces       A vector of faces that will be used to query the training data
 * 
 */
void classifierThreshold(const char* resultsPath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<pair<string, VectorXf> > trainingFaces, vector<pair<string, VectorXf> > queryFaces)
{
    // Perform PCA dimensionality reduction

    float eigenValuesSum = eigenvalues.sum();
    float currentEigenTotal = 0;
    int count;
    char fileName[100];

    //Find the number of vectors to preserve PCA_PERCENTAGE of the information
    for(count = 0; currentEigenTotal / eigenValuesSum < PCA_PERCENTAGE && count < eigenvalues.rows(); count++)
    {
        currentEigenTotal += eigenvalues.row(count)(0);
    }

    cout << "Reducing Dimensionality from " << eigenfaces.cols() << " to " << count << "!" << endl;

    MatrixXf reducedEigenFaces(averageFace.rows(), count);

    reducedEigenFaces = eigenfaces.block(0,0,averageFace.rows(),count);
    
    //Project the faces onto the reduced eigenfaces

    vector<pair<string, VectorXf> > projectedTrainingFaces, projectedQueryFaces;
    
    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        pair<string, VectorXf> temp(queryFaces[i].first, projectOntoEigenspace(queryFaces[i].second, averageFace, reducedEigenFaces));
        projectedQueryFaces.push_back(temp);
    }    

    cout << "Reduced!" << endl;

    VectorXf projQueryFace;
    int TPCount, FPCount;
    TPCount = FPCount = 0;

    pair<int, int> temp(0,0);

    vector< pair<int, int> > counts(1800, temp); //first = TP count & second = FP count

    // Iterate through each query face and see if it can be classified correctly

    for(unsigned int i = 0; i < projectedQueryFaces.size(); i++)
    {
        cout << "\rQuery Face: " << i;
        projQueryFace = projectedQueryFaces[i].second;
        vector< pair<string, float> > queryPairs; //Pair is training image id (string) and distance (float)

        for(unsigned int t = 0; t < trainingFaces.size(); t++)
        {
            pair<string, float> newPair(trainingFaces[t].first, distanceInFaceSpace(projQueryFace, trainingFaces[t].second));
            queryPairs.push_back(newPair);
        }

        sort(queryPairs.begin(), queryPairs.end(), cmp);
        cout << "\t" << queryPairs[0].second << endl;


        //For High Images 380, 1500, +=5
        //For Low Images 50, 600, += 2

        for(int threshold = 50; threshold < 600; threshold+=2)
        {
            //Our best match (from the allowed subjects) is less than the threshold 
            //So we have a positive result
            if(queryPairs[0].second <= threshold)
            {
                //Check if its true positive or false positive
                if(atoi(projectedQueryFaces[i].first.c_str()) <= 93)
                {
                    //True positive - We have a query face that is in the training sample (Non-intruder allowed access)
                    counts[threshold].first++;
                }
                else
                {
                    //False positive - We have a query face that is NOT in the training sample (Intruder)
                    counts[threshold].second++;
                }
            }
        }

    }

    sprintf(fileName, "%s-%i.txt", resultsPath, (int)(PCA_PERCENTAGE*100));
    ofstream output;

    output.open(fileName);

    //Print out data for ROC curve

    for(int threshold = 50; threshold < 600; threshold+=2)
    {
        float TPRate = counts[threshold].first / (float)trainingFaces.size();
        float FPRate = counts[threshold].second / (float)(queryFaces.size() - trainingFaces.size());

        output << threshold << "\t" << TPRate<< "\t" << FPRate << endl;
    }

    output.close();

}
    
