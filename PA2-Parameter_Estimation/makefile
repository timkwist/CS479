CXX=g++
INC=../
CXXFLAGS=-g -Wall -pedantic -std=c++11 -I$(INC)
OBJ = main.o SampleGenerator.o BayesClassifier.o boxmuller.o MLE.o

BayesClassifier.exe: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf *.o
	rm BayesClassifier
