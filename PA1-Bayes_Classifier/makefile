CXX=g++
INC=../
CXXFLAGS=-g -Wall -pedantic -std=c++11 -I$(INC)
OBJ = main.o BayesClassifier.o boxmuller.o

BayesClassifier: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf *.o
	rm BayesClassifier
