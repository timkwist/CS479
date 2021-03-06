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

    // Part A

    readInFaces("./fa_H", trainingFaces);
    readInFaces("./fb_H", queryFaces);


    cout << "Reading in saved faces, if possible" << endl;
    if(readSavedFaces(averageFace, eigenfaces, eigenvalues, "fa_H") == false) // faces haven't been computed yet
    {
    	cout << "No saved faces available, computing faces instead" << endl;
        
    	computeEigenFaces(trainingFaces, averageFace, eigenfaces, eigenvalues, "fa_H");
    }

    normalizeEigenFaces(eigenfaces);

    // Print average face
    writeFace(averageFace, "averageFace.pgm");


    runClassifier("N-Results/NData", averageFace, eigenfaces, eigenvalues, trainingFaces, queryFaces);


    // Print top 10 eigenvalues
    char faceFileName[100];
    for(int i = 0; i < 10; i++)
    {
        sprintf(faceFileName, "Part-AlargestFace%i.pgm", i + 1);
        writeFace(eigenfaces.col(i), faceFileName);
    }

    // Print top 10 eigenvalues
    for(int i = eigenfaces.cols() - 1; i > eigenfaces.cols() - 1 - 10; i--)
    {
        sprintf(faceFileName, "Part-AsmallestFace%i.pgm", i - eigenfaces.cols() + 2);
        writeFace(eigenfaces.col(i), faceFileName);
    }

    // Part B

    trainingFaces.clear();
    queryFaces.clear();

    readInFaces("./fa2_H", trainingFaces);
    readInFaces("./fb_H", queryFaces);

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

    //Part C

    trainingFaces.clear();
    queryFaces.clear();

    readInFaces("./fa_L", trainingFaces);
    readInFaces("./fb_L", queryFaces);

    cout << "Reading in saved faces for part c, if possible" << endl;
    if(readSavedFaces(averageFace, eigenfaces, eigenvalues, "fa_L") == false) // faces haven't been computed yet
    {
        cout << "No saved faces available, computing faces instead" << endl;
        
        computeEigenFaces(trainingFaces, averageFace, eigenfaces, eigenvalues, "fa2_H");
    }

    normalizeEigenFaces(eigenfaces);

    writeFace(averageFace, "averageFace-PartC.pgm");

    PCA_PERCENTAGE = atof(argv[1]);

    runClassifier("PartC-Results/CData", averageFace, eigenfaces, eigenvalues, trainingFaces, queryFaces);

    // Print top 10 eigenvalues
    char faceFileName[100];
    for(int i = 0; i < 10; i++)
    {
        sprintf(faceFileName, "PartC-largestFace%i.pgm", i + 1);
        writeFace(eigenfaces.col(i), faceFileName);
    }

    // Print top 10 eigenvalues
    for(int i = eigenfaces.cols() - 1; i > eigenfaces.cols() - 1 - 10; i--)
    {
        sprintf(faceFileName, "PartC-smallestFace%i.pgm", i - eigenfaces.cols() + 2);
        writeFace(eigenfaces.col(i), faceFileName);
    }


    //Part D

    trainingFaces.clear();
    queryFaces.clear();

    readInFaces("./fa2_L", trainingFaces);
    readInFaces("./fb_L", queryFaces);

    cout << "Reading in saved faces for part b, if possible" << endl;
    if(readSavedFaces(averageFace, eigenfaces, eigenvalues, "fa2_L") == false) // faces haven't been computed yet
    {
        cout << "No saved faces available, computing faces instead" << endl;
        
        computeEigenFaces(trainingFaces, averageFace, eigenfaces, eigenvalues, "fa2_L");
    }

    normalizeEigenFaces(eigenfaces);

    writeFace(averageFace, "averageFace-PartD.pgm");

    PCA_PERCENTAGE = .95;

    classifierThreshold("PartD-CResults/CData", averageFace, eigenfaces, eigenvalues, trainingFaces, queryFaces);

}

/**
 * Normalizes each eigenface in a matrix.
 * 
 * @param eigenfaces  A matrix of eigen faces to normalize
 */
void normalizeEigenFaces(MatrixXf &eigenfaces)
{
    for(int i = 0; i < eigenfaces.cols(); i++)
    {
        eigenfaces.col(i).normalize();  
    }
}



