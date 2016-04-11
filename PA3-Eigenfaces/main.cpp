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


float PCA_PERCENTAGE    = .80;


/* External methods */
int readImageHeader(char[], int&, int&, int&, bool&);
int readImage(char[], ImageType&);
int writeImage(char[], ImageType&);

void readInFaces(const char *path, vector<pair<string, VectorXf> > &faces);
bool readSavedFaces(VectorXf &averageFace, MatrixXf &eigenfaces, VectorXf &eigenvalues, const char *path);
void writeFace(VectorXf theFace, char *fileName);
bool fileExists(const char *filename);

void computeEigenFaces(vector<pair<string, VectorXf> > trainingFaces, VectorXf &averageFace, MatrixXf &eigenfaces, VectorXf &eigenvalues, const char *path);
float distanceInFaceSpace(VectorXf originalFace, VectorXf newFace);
VectorXf projectOntoEigenspace(VectorXf newFace, VectorXf averageFace, MatrixXf eigenfaces);
bool amongNMostSimilarFaces(vector<pair<string, float> > similarFaces, int N, string searchID);

void runClassifier(const char* resultsPath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<pair<string, VectorXf> > trainingFaces, vector<pair<string, VectorXf> > queryFaces);
void classifierThreshold(const char* resultsPath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<pair<string, VectorXf> > trainingFaces, vector<pair<string, VectorXf> > queryFaces);


/* Internal methods */
void normalizeEigenFaces(MatrixXf &eigenfaces);

int main(int argc, char* argv[])
{

    vector<pair<string, VectorXf> > trainingFaces, queryFaces;
    MatrixXf eigenfaces;
    VectorXf eigenvalues;
    VectorXf averageFace;

    if(argc < 2)
    {
        cout << "Improper Usage! Needs one argument for Percentage \n";
        return 1;
    }

    PCA_PERCENTAGE = atof(argv[1]);


    /**
     * TRAINING MODE
     */
    //================================================
    // Compute Average Face and Eigenfaces and Eigenvalues
    //================================================

    // readInFaces("./fa_H", trainingFaces);
    // readInFaces("./fb_H", queryFaces);


    // cout << "Reading in saved faces, if possible" << endl;
    // if(readSavedFaces(averageFace, eigenfaces, eigenvalues, "fa_H") == false) // faces haven't been computed yet
    // {
    // 	cout << "No saved faces available, computing faces instead" << endl;
        
    // 	computeEigenFaces(trainingFaces, averageFace, eigenfaces, eigenvalues, "fa_H");
    // }

    // normalizeEigenFaces(eigenfaces);

    // // Print average face
    // writeFace(averageFace, "averageFace.pgm");


    // runClassifier("N-Results/NData", averageFace, eigenfaces, eigenvalues, trainingFaces, queryFaces);


    // Print top 10 eigenvalues
    // char faceFileName[100];
    // for(int i = 0; i < 10; i++)
    // {
    //     sprintf(faceFileName, "largestFace%i.pgm", i + 1);
    //     writeFace(eigenfaces.col(i), faceFileName);
    // }

    // // Print top 10 eigenvalues
    // for(int i = eigenfaces.cols() - 1; i > eigenfaces.cols() - 1 - 10; i--)
    // {
    //     sprintf(faceFileName, "smallestFace%i.pgm", i - eigenfaces.cols() + 2);
    //     writeFace(eigenfaces.col(i), faceFileName);
    // }

    // trainingFaces.clear();
    // queryFaces.clear();

    readInFaces("./fa2_H", trainingFaces);
    readInFaces("./fa_H", queryFaces);

    cout << "Reading in saved faces for part b, if possible" << endl;
    if(readSavedFaces(averageFace, eigenfaces, eigenvalues, "fa2_H") == false) // faces haven't been computed yet
    {
        cout << "No saved faces available, computing faces instead" << endl;
        
        computeEigenFaces(trainingFaces, averageFace, eigenfaces, eigenvalues, "fa2_H");
    }

    normalizeEigenFaces(eigenfaces);

    writeFace(averageFace, "averageFace-PartB.pgm");

    PCA_PERCENTAGE = .95;

    classifierThreshold("C-Results/CData", averageFace, eigenfaces, eigenvalues, trainingFaces, queryFaces);


/*

    //================================================
    // Interactive: Decide how many faces to keep
    //================================================

    //================================================
    // Project training face image onto eigenspace
    //================================================

    //================================================
    // Compute representation in this space
    //================================================

    //================================================
    // Store coefficients, average face, and eigenfaces
    //================================================

    /**
     * TESTING MODE
     */
    //================================================
    // Read in coefficients, average face, eigenfaces
    //================================================

    //================================================
    // Evaluate face recognition performance from test set
    // - Project onto eigenspace
    // - compute coefficients
    // - Find closest match in face space
    //================================================



}


void normalizeEigenFaces(MatrixXf &eigenfaces)
{
    for(int i = 0; i < eigenfaces.cols(); i++)
    {
        eigenfaces.col(i).normalize();  
    }
}



