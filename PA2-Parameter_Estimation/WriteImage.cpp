#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

#include "image.h"

void writeImage(char fname[], ImageType& image)
{
 int i, j;
 int N, M, Q;
 unsigned char *charImage;
 ofstream ofp;

 image.getImageInfo(N, M, Q);

 charImage = (unsigned char *) new unsigned char [3*M*N];

 RGB val;

 for(i=0; i<N; i++)
   for(j=0; j<3*M; j+=3) {
     image.getPixelVal(i, j/3, val);
     charImage[i*3*M+j]=(unsigned char)val.r;
     charImage[i*3*M+j+1]=(unsigned char)val.g;
     charImage[i*3*M+j+2]=(unsigned char)val.b;
   }

 ofp.open(fname, ios::out | ios::binary);

 if (!ofp) {
   cout << "Can't open file: " << fname << endl;
   exit(1);
 }

 ofp << "P6" << endl;
 ofp << M << " " << N << endl;
 ofp << Q << endl;

 ofp.write( reinterpret_cast<char *>(charImage), (3*M*N)*sizeof(unsigned char));

 if (ofp.fail()) {
   cout << "Can't write image " << fname << endl;
   exit(0);
 }

 ofp.close();

 delete [] charImage;
}
