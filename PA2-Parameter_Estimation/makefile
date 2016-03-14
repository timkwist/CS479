CXX=g++
INC=../
CXXFLAGS=-g -Wall -pedantic -std=c++11 -I$(INC)
OBJ = SampleGenerator.o BayesClassifier.o boxmuller.o MLE.o image.o ReadImageHeader.o ReadImage.o WriteImage.o

MLE-1-2.exe: $(OBJ) main-part1-2.o
	$(CXX) $(CXXFLAGS) -o $@ $^
	
MLE-3.exe: $(OBJ) main-part3.o
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf *.o
	rm BayesClassifier
