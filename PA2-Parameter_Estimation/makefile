CXX=g++
INC=../
CXXFLAGS=-g -Wall -pedantic -std=c++11 -I$(INC)
OBJ = testmain.o SampleGenerator.o BayesClassifier.o boxmuller.o MLE.o image.o ReadImageHeader.o ReadImage.o WriteImage.o

MLE.exe: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf *.o
	rm BayesClassifier
