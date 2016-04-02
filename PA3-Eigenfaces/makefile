CXX=g++
INC=../
CXXFLAGS=-g -Wall -pedantic -std=c++11 -I$(INC)
OBJ = image.o ReadImageHeader.o ReadImage.o

Eigenfaces.exe: $(OBJ) main.o
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf *.o
	rm -rf *.exe