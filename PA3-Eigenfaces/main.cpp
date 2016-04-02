#include <iostream>
#include <Eigen/Dense>
#include <dirent.h>
#include <vector>
#include <cstdlib>

#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

#include "image.h"

int readImageHeader(char[], int&, int&, int&, bool&);
int readImage(char[], ImageType&);
vector<VectorXi> readInTrainingFaces(const char *path, vector<VectorXi> &trainingFaces);

int main()
{
    vector<VectorXi> trainingFaces;
    VectorXf averageFace;
    /**
     * TRAINING MODE
     */
    //================================================
    // Read in Training Faces
    //================================================
    readInTrainingFaces("./fb_H", trainingFaces);
    //================================================
    // Compute Average Face and Eigenfaces
    //================================================
    averageFace = VectorXf(trainingFaces[0].rows());
    averageFace.fill(0);
    for(auto it = trainingFaces.begin(); it != trainingFaces.end(); it++)
    {
        averageFace += (*it).cast<float>();
    }
    averageFace /= trainingFaces.size();
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