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
            pair<string, float> newPair(trainingFaces[t].first, distanceInFaceSpace(projQueryFace, projectedTrainingFaces[t].second));
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

void classifierThreshold(const char* resultsPath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<pair<string, VectorXf> > trainingFaces, vector<pair<string, VectorXf> > queryFaces)
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

    // for(unsigned int i = 0; i < trainingFaces.size(); i++)
    // {
    //     pair<string, VectorXf> temp(trainingFaces[i].first, projectOntoEigenspace(trainingFaces[i].second, averageFace, reducedEigenFaces));
    //     projectedTrainingFaces.push_back(temp);
    // }
    
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
            else
            {
                //Check if its false negative or true negative
                // if(atoi(projectedQueryFaces[i].first.c_str()) <= 93)
                // {
                //     //False negative - We have a query face that is in the training sample (Non-intruder denied acccess)
                // }
                // else
                // {
                //     //True negative - We have a query face that is NOT in the training sample (Intruder denied access)
                // }
            }
        }

    }

    sprintf(fileName, "%s-%i.txt", resultsPath, (int)(PCA_PERCENTAGE*100));
    ofstream output;

    output.open(fileName);

    //50 non intruders and 817 intruders

    for(int threshold = 50; threshold < 600; threshold+=2)
    {
        float TPRate = counts[threshold].first / (float)trainingFaces.size();
        float FPRate = counts[threshold].second / (float)(queryFaces.size() - trainingFaces.size());

        output << threshold << "\t" << TPRate<< "\t" << FPRate << endl;
    }

    output.close();

    //Check each query face against the training set, if any comparisons are below the threshold then they are allowed in

}
    
