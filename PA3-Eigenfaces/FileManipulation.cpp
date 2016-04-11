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

/* External methods */
int readImageHeader(char[], int&, int&, int&, bool&);
int readImage(char[], ImageType&);
int writeImage(char[], ImageType&);

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

/**
 * Returns whether or not the given file exists (IE, can be opened/read)
 * 
 * @param  filename Name of file that is desired to be opened
 * @return          True if the file does exists, false otherwise.
 */
bool fileExists(const char *filename)
{
    ifstream ifile(filename);
    return ifile;
}

/**
 * Checks if the averageface, eigenfaces, and eigenvalues for the given path
 * exist.
 * If they do exist, reads them in and returns them via the reference parameters.
 *
 * NOTE: Reads files in the order of
 * - Eigenfaces
 * - Eigenvalues
 * - Average face
 * Will attempt to read each of these in individually in their given order
 * (IE, only checks if a given file exists after previous files have been read in)
 * 
 * @param  averageFace Empty vector that will be filled if the file exists
 * @param  eigenfaces  Empty matrix that will be filled if the file exists
 * @param  eigenvalues Empty vector that will be filled if the file exists
 * @param  path        Path of the files to search for
 * @return             Returns true if all three files are available; returns false otherwise
 */
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

/**
 * Reads in the face image files from the desired path
 * and returns faces by reference.
 *
 * In the return vector, the string that is recorded is the 5 digit ID associated with
 * that face image.
 * 
 * @param path  Desired path to get faces from
 * @param faces Empty vector of pair<string, VectorXf> that will be returned with
 *              all faces in the given path CONCATENATED to the end of the vector
 */
void readInFaces(const char *path, vector<pair<string, VectorXf> > &faces)
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
            faces.push_back(pair<string, VectorXf>(string(ent->d_name, 5), currentFace));
        }
        closedir (dir);
    }
}

/**
 * Writes the given face to the given file name / path
 * Currently only supports 20x16 and 60x48 images, will crash
 * otherwise.
 * 
 * @param theFace  The desired face to write the file
 * @param fileName Desired file name to write to
 */
void writeFace(VectorXf theFace, char *fileName)
{
    int rows, cols, levels;

    if(theFace.rows() == 320)
    {
        rows = 20;
        cols = 16;
    }
    else if(theFace.rows() == 2880)
    {
        rows = 60;
        cols = 48;
    }

    levels = 255;

    ImageType theImage(rows, cols, levels);

    float min, max, val;
    min = theFace.minCoeff();
    max = theFace.maxCoeff();

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            val = (theFace[i*cols + j] - min) / (max - min);
            theImage.setPixelVal(i, j, val*255);
        }
    }

    writeImage(fileName, theImage);
}

/**
 * Computes the eigenfaces, eigenvalues, and average face of the eigenspace defined by the passed in
 * training faces.
 * Additionally, saves the eigenfaces, eigenvalues, and average face to file so that they can be read
 * in at a later date.
 * 
 * @param trainingFaces Vector of training faces that make of the eigenspace
 * @param averageFace   Empty vector that will be filled by the average face and returned by reference
 * @param eigenfaces    Empty matrix that will be filled by all eigenfaces and returned by reference
 * @param eigenvalues   Empty vector that will be filled by all eigenvalues and returned by reference
 * @param path          Desired path/name to save the average face, eigenfaces, and eigenvalues to
 */
void computeEigenFaces(vector<pair<string, VectorXf> > trainingFaces, VectorXf &averageFace, MatrixXf &eigenfaces, VectorXf &eigenvalues, const char *path)
{
    char fileName[100];
    EigenSolver<MatrixXf> solver;
    MatrixXf A;
    ofstream output;

    MatrixXf eigenVectors;


    averageFace = VectorXf(trainingFaces[0].second.rows());
    averageFace.fill(0);
    for(auto it = trainingFaces.begin(); it != trainingFaces.end(); it++)
    {
        averageFace += (*it).second;
    }
    averageFace /= trainingFaces.size();

    sprintf(fileName, "%s-avg-binary.dat", path);

    Eigen::write_binary(fileName, averageFace);

    A = MatrixXf(averageFace.rows(), trainingFaces.size());
    for(vector<VectorXf>::size_type i = 0; i < trainingFaces.size(); i++)
    {
        A.col(i) = trainingFaces[i].second - averageFace;
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