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


const float PCA_PERCENTAGE = .80;

/* Helper Methods */
namespace Eigen
{
    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix)
    {
        std::ofstream out(filename,ios::out | ios::binary | ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }

    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix)
    {
        std::ifstream in(filename,ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }
}


/* External methods */
int readImageHeader(char[], int&, int&, int&, bool&);
int readImage(char[], ImageType&);
int writeImage(char[], ImageType&);

/* Internal methods */
vector<VectorXf> readInFaces(const char *path, vector<VectorXf> &faces);
void computeEigenFaces(vector<VectorXf> trainingFaces, VectorXf &averageFace, MatrixXf &eigenfaces, VectorXf &eigenvalues, const char *path);
float distanceInFaceSpace(VectorXf originalFace, VectorXf newFace);
void writeFace(VectorXf theFace, char *fileName);
bool readSavedFaces(VectorXf &averageFace, MatrixXf &eigenfaces, VectorXf &eigenvalues, const char *path);
bool fileExists(const char *filename);
VectorXf projectOntoEigenspace(VectorXf newFace, VectorXf averageFace, MatrixXf eigenfaces);

void runClassifier(const char* resultsPath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<VectorXf> trainingFaces, vector<VectorXf> queryFaces);

void normalizeEigenFaces(MatrixXf &eigenfaces);

int main()
{

    vector<VectorXf> trainingFaces, queryFaces;
    MatrixXf eigenfaces;
    VectorXf eigenvalues;
    VectorXf averageFace;
    /**
     * TRAINING MODE
     */
    //================================================
    // Compute Average Face and Eigenfaces and Eigenvalues
    //================================================

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


    runClassifier("Results.txt", averageFace, eigenfaces, eigenvalues, trainingFaces, queryFaces);


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

    // vector<VectorXf> newFaces;

    // newFaces.push_back(projectOntoEigenspace(trainingFaces[152], averageFace, eigenfaces));

    // writeFace(newFaces[0], "testing.pgm");

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

vector<VectorXf> readInFaces(const char *path, vector<VectorXf> &faces)
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (path)) != NULL)
    {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL)
        {
            if(ent->d_name[0] == '.')
                continue;
            bool type;
            int rows, cols, levels;
            VectorXf currentFace;
            char name[50] = "";
            strcat(name, path);
            strcat(name, "/");
            strcat(name, ent->d_name);

            // read training images' headers
            readImageHeader(name, rows, cols, levels, type);
            // allocate memory for the image array
            ImageType currentImage(rows, cols, levels);

            // read image
            readImage(name, currentImage);

            currentFace = VectorXf(rows*cols);
            for(int i = 0; i < rows; i++)
            {
                for(int j = 0; j < cols; j++)
                {
                    int t = 0; // Temp placeholder int to get pixel val
                    currentImage.getPixelVal(i, j, t);
                    currentFace[i*cols + j] = t;
                }
            }
            faces.push_back(currentFace);
        }
        closedir (dir);
    }

    return faces;
}

void writeFace(VectorXf theFace, char *fileName)
{
    int rows, cols, levels;
    bool type;
    readImageHeader("fb_H/00001_930831_fb_a.pgm", rows, cols, levels, type);

    ImageType theImage(rows, cols, levels);

    float min, max, val;
    // float mean;
    min = theFace.minCoeff();
    max = theFace.maxCoeff();
    // mean = theFace.mean();

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            val = (theFace[i*cols + j] - min) / (max - min);
            theImage.setPixelVal(i, j, val*255);
        }
        // cout << endl;
    }

    writeImage(fileName, theImage);

}

void computeEigenFaces(vector<VectorXf> trainingFaces, VectorXf &averageFace, MatrixXf &eigenfaces, VectorXf &eigenvalues, const char *path)
{
    char fileName[100];
    EigenSolver<MatrixXf> solver;
    MatrixXf A;
    ofstream output;

    MatrixXf eigenVectors;


    averageFace = VectorXf(trainingFaces[0].rows());
    averageFace.fill(0);
    for(auto it = trainingFaces.begin(); it != trainingFaces.end(); it++)
    {
        averageFace += (*it);
    }
    averageFace /= trainingFaces.size();

    sprintf(fileName, "%s-avg-binary.dat", path);

    Eigen::write_binary(fileName, averageFace);

    A = MatrixXf(averageFace.rows(), trainingFaces.size());
    for(vector<VectorXf>::size_type i = 0; i < trainingFaces.size(); i++)
    {
        A.col(i) = trainingFaces[i] - averageFace;
    }
    eigenVectors = MatrixXf(trainingFaces.size(), trainingFaces.size());
    eigenVectors = A.transpose()*A;
    solver.compute(eigenVectors, /* Compute eigenvectors = */ true);

    eigenfaces = MatrixXf(averageFace.rows(), trainingFaces.size());

    eigenfaces = A * solver.eigenvectors().real();

    eigenvalues = VectorXf(eigenfaces.cols());

    eigenvalues = solver.eigenvalues().real();

    // Eigen saves eigenvectors in a strange format, so writing them to a file
    // and then reading them back in saves them in a more "expected" format
    sprintf(fileName, "%s-binary.dat", path);
    Eigen::write_binary(fileName, eigenfaces);
    Eigen::read_binary(fileName,eigenfaces);

    sprintf(fileName, "%s-values-binary.dat", path);

    Eigen::write_binary(fileName, eigenvalues);

    // Save eigenvectors to text file as well for viewing outside program
    sprintf(fileName, "%s-EigenVectors.txt", path);
    output.open(fileName);
    output << eigenfaces;
    output.close();
}

bool readSavedFaces(VectorXf &averageFace, MatrixXf &eigenfaces, VectorXf &eigenvalues, const char *path)
{
    char fileName[100];

    sprintf(fileName, "%s-binary.dat", path);

    if(fileExists(fileName))
    {
        Eigen::read_binary(fileName, eigenfaces);
    }
    else
    {
    	return false; 
    }

    sprintf(fileName, "%s-values-binary.dat", path);

    if(fileExists(fileName))
    {
        Eigen::read_binary(fileName, eigenvalues);
    }
    else
    {
        return false;
    }

    sprintf(fileName, "%s-avg-binary.dat", path);

    if(fileExists(fileName))
    {
        Eigen::read_binary(fileName, averageFace);
    }
    else
    {
    	return false;
    }



    return true;
}

float distanceInFaceSpace(VectorXf originalFace, VectorXf newFace)
{
	return (originalFace - newFace).norm();
}

bool fileExists(const char *filename)
{
    ifstream ifile(filename);
    return ifile;
}


VectorXf projectOntoEigenspace(VectorXf newFace, VectorXf averageFace, MatrixXf eigenfaces)
{
	vector<float> faceCoefficients;
	VectorXf normalizedFace = newFace - averageFace;
	VectorXf projectedFace(averageFace.rows());
	projectedFace.fill(0);
    normalizedFace.normalize();
	for(int i = 0; i < eigenfaces.cols(); i++)
	{
		float a = (eigenfaces.col(i).transpose() * normalizedFace)(0,0);
		faceCoefficients.push_back(a);
		projectedFace += (faceCoefficients[i] * eigenfaces.col(i));

	}
    return projectedFace;
}

void normalizeEigenFaces(MatrixXf &eigenfaces)
{
    for(int i = 0; i < eigenfaces.cols(); i++)
    {
        eigenfaces.col(i).normalize();  
    }
}

void runClassifier(const char* resultsPath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<VectorXf> trainingFaces, vector<VectorXf> queryFaces)
{
    float eigenValuesSum = eigenvalues.sum();
    float currentEigenTotal = 0;
    int count;

    for(count = 0; currentEigenTotal / eigenValuesSum < PCA_PERCENTAGE && count < eigenvalues.rows(); count++)
    {
        currentEigenTotal += eigenvalues.row(count)(0);
    }

    cout << "Reducing Dimensionality from " << eigenfaces.cols() << " to " << count << "!" << endl;

    MatrixXf reducedEigenFaces(averageFace.rows(), count);

    reducedEigenFaces = eigenfaces.block(0,0,averageFace.rows(),count);
    
    vector<VectorXf> projectedTrainingFaces, projectedQueryFaces;

    for(unsigned int i = 0; i < trainingFaces.size(); i++)
    {
        trainingFaces[i].normalize();
        projectedTrainingFaces.push_back(projectOntoEigenspace(trainingFaces[i], averageFace, reducedEigenFaces));
    }
    
    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        queryFaces[i].normalize();
        projectedQueryFaces.push_back(projectOntoEigenspace(queryFaces[i], averageFace, reducedEigenFaces));
    }    

    cout << "Reduced!" << endl;

    VectorXf projQueryFace;

    vector< pair<int, float> > queryPairs; //Pair is training image index (int) and distance (float)

    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        cout << "Query Face: " << i << endl;
        projQueryFace = projectedQueryFaces[i];

        for(unsigned int t = 0; t < trainingFaces.size(); t++)
        {
            pair<int, float> newPair(t, distanceInFaceSpace(projQueryFace, trainingFaces[t]));

            if(newPair.second < 1)
                cout << newPair.first << "  " << newPair.second << endl;

            queryPairs.push_back(newPair);
        }
    }



    
}