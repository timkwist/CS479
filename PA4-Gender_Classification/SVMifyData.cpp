#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "BayesClassifier.h"
#include "MLE.h"
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>


using namespace Eigen;
using namespace std;

void saveNewFile(char* fileName) {
	string inFile = "./genderdata/16_20/";
	string curData = "", curLabel = "";
	ifstream finData, finLabel;
	istringstream iss;
	ofstream fout(fileName);
	float curFloat = 0.0;
	int counter = 1, numEig = 0;

	inFile += fileName;
	finData.open(inFile);
	inFile = "./genderdata/16_20/T";
	inFile += fileName;
	finLabel.open(inFile);

	while(numEig < 30 && getline(finData, curData) && getline(finLabel, curLabel)) {
		fout << curLabel;
		iss.clear();
		iss.str(curData);
		counter = 1;
		while(iss >> curFloat) {
			fout << " " << counter << ":" << curFloat;
			counter++;
		}
		fout << endl;
		numEig++;
	}
	finData.close();
	finLabel.close();
	fout.close();

}

int main()
{
	char fileName[256];
	for(int i = 1; i <= 3; i++) {
		sprintf(fileName, "trPCA_0%i-new.txt", i);
		saveNewFile(fileName);
		sprintf(fileName, "TtrPCA_0%i-new.txt", i);
		saveNewFile(fileName);
		sprintf(fileName, "tsPCA_0%i-new.txt", i);
		saveNewFile(fileName);
		sprintf(fileName, "TtsPCA_0%i-new.txt", i);
		saveNewFile(fileName);
		sprintf(fileName, "valPCA_0%i-new.txt", i);
		saveNewFile(fileName);
		sprintf(fileName, "TvalPCA_0%i-new.txt", i);
		saveNewFile(fileName);
	}
	return 0;
}