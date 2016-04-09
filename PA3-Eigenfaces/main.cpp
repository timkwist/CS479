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
vector<VectorXi> readInTrainingFaces(const char *path, vector<VectorXi> &trainingFaces);
void computeEigenFaces(vector<VectorXi> trainingFaces, VectorXf &averageFace, MatrixXf &eigenfaces, const char *path);
float distanceInFaceSpace(VectorXi originalFace, VectorXi newFace);
void writeFace(VectorXf theFace, char *fileName);
bool readSavedFaces(VectorXf &averageFace, MatrixXf &eigenfaces, const char *path);
bool fileExists(const char *filename);
VectorXf projectOntoEigenspace(VectorXi newFace, VectorXi averageFace, MatrixXf eigenfaces);


int main()
{

    vector<VectorXi> trainingFaces;
    MatrixXf eigenfaces;
    VectorXf averageFace;
    /**
     * TRAINING MODE
     */
    //================================================
    // Compute Average Face and Eigenfaces
    //================================================
    cout << "Reading in saved faces, if possible" << endl;
    if(readSavedFaces(averageFace, eigenfaces, "fb_H") == false) // faces haven't been computed yet
    {
    	cout << "No saved faces available, computing faces instead" << endl;
        readInTrainingFaces("./fb_H", trainingFaces);
    	computeEigenFaces(trainingFaces, averageFace, eigenfaces, "fb_H");
    }


    writeFace(eigenfaces.col(0), "testing2.pgm");


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

vector<VectorXi> readInTrainingFaces(const char *path, vector<VectorXi> &trainingFaces)
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
            VectorXi currentFace;
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

            currentFace = VectorXi(rows*cols);
            for(int i = 0; i < rows; i++)
            {
                for(int j = 0; j < cols; j++)
                {
                    int t = 0; // Temp placeholder int to get pixel val
                    currentImage.getPixelVal(i, j, t);
                    currentFace[i*cols + j] = t;
                }
            }
            trainingFaces.push_back(currentFace);
        }
        closedir (dir);
    }

    return trainingFaces;
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

void computeEigenFaces(vector<VectorXi> trainingFaces, VectorXf &averageFace, MatrixXf &eigenfaces, const char *path)
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
        averageFace += (*it).cast<float>();
    }
    averageFace /= trainingFaces.size();

    sprintf(fileName, "%s-avg-binary.dat", path);

    Eigen::write_binary(fileName, averageFace);

    A = MatrixXf(averageFace.rows(), trainingFaces.size());
    for(vector<VectorXi>::size_type i = 0; i < trainingFaces.size(); i++)
    {
        A.col(i) = trainingFaces[i].cast<float>() - averageFace;
    }
    eigenVectors = MatrixXf(trainingFaces.size(), trainingFaces.size());
    eigenVectors = A.transpose()*A;
    solver.compute(eigenVectors, /* Compute eigenvectors = */ true);

    eigenfaces = MatrixXf(averageFace.rows(), trainingFaces.size());

    eigenfaces = A * solver.eigenvectors().real();

    // Eigen saves eigenvectors in a strange format, so writing them to a file
    // and then reading them back in saves them in a more "expected" format
    sprintf(fileName, "%s-binary.dat", path);
    Eigen::write_binary(fileName, eigenfaces);
    Eigen::read_binary(fileName,eigenfaces);

    // Save eigenvectors to text file as well for viewing outside program
    sprintf(fileName, "%s-EigenVectors.txt", path);
    output.open(fileName);
    output << eigenfaces;
    output.close();
}

bool readSavedFaces(VectorXf &averageFace, MatrixXf &eigenfaces, const char *path)
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

float distanceInFaceSpace(VectorXi originalFace, VectorXi newFace)
{
	return (originalFace - newFace).norm();
}

bool fileExists(const char *filename)
{
    ifstream ifile(filename);
    return ifile;
}

VectorXf projectOntoEigenspace(VectorXi newFace, VectorXi averageFace, MatrixXf eigenfaces)
{
	vector<float> faceCoefficients;
	VectorXf normalizedFace = newFace.cast<float>() - averageFace.cast<float>();
	VectorXf transposedFace = normalizedFace.transpose();
	VectorXf projectedFace(averageFace.rows());
	projectedFace.fill(0);
	for(int i = 0; i < eigenfaces.cols(); i++)
	{
		float a = (transposedFace * eigenfaces.col(i))(0,0);
		faceCoefficients.push_back(a);
		projectedFace += (faceCoefficients[i] * normalizedFace);

	}

    return projectedFace;
}