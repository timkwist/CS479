CXX=g++
INC=../
CXXFLAGS=-g -Wall -pedantic -std=c++11 -I$(INC)
OBJ = image.o ReadImageHeader.o ReadImage.o WriteImage.o FileManipulation.o Classification.o

Eigenfaces.exe: $(OBJ) main.o
	$(CXX) $(CXXFLAGS) -o $@ $^

test.exe: $(OBJ) test.o
	$(CXX) $(CXXFLAGS) -o $@ $^
clean:
	rm -rf *.o
	rm -rf *.exe
