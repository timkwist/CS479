CXX=g++
INC=../
CXXFLAGS=-g -Wall -pedantic -std=c++11 -I$(INC)
OBJ = BayesClassifier.o boxmuller.o MLE.o

GC.exe: $(OBJ) main.o
	$(CXX) $(CXXFLAGS) -o $@ $^
	
clean:
	rm -rf *.o
	rm -rf *.exe
