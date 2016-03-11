#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"
#include "SampleGenerator.h"
#include "MLE.h"
#include <cstdlib>

using namespace std;
using namespace Eigen;

#include "image.h"

int readImageHeader(char[], int&, int&, int&, bool&);
int readImage(char[], ImageType&);
int writeImage(char[], ImageType&);

int main(int argc, char *argv[])
{
	int i, j; 
	int M, N, Q;
	bool type;
	RGB val;
	vector<Vector2f> sampleData;

 	// read image header
	readImageHeader("ref1.ppm", N, M, Q, type);
	readImageHeader("train1.ppm", N, M, Q, type);

 	// allocate memory for the image array

	ImageType ref(N, M, Q);
	ImageType train(N, M, Q);
	ImageType newImg(N, M, Q);

 	// read image
	readImage("ref1.ppm", ref);
	readImage("train1.ppm", train);

	

 	// threshold image 

	for(i=0; i<N; i++)
	{
		for(j=0; j<M; j++) 
		{
			ref.getPixelVal(i, j, val);
			if(val.r != 0 && val.g != 0 && val.b != 0)
			{
				train.getPixelVal(i, j, val);
				float total = val.r + val.g + val.b;
				float newR = (float)val.r / total;
				float newG = (float)val.g / total;
				float newB = (float)val.b / total;
				newImg.setPixelVal(i, j, RGB(255*newR, 255*newG, 255*newB));
				sampleData.push_back(Vector2f(newR, newG));
			}
			else
			{
				newImg.setPixelVal(i, j, RGB(0,0,0));
			}
		}
	}

	writeImage("test.ppm", newImg);

	return (1);
}