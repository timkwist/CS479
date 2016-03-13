#ifndef RGB_H
#define RGB_H

// an example -- you would need to add more functions

struct RGB {
	RGB()
	{
		r = g = b = 0;
	}
  RGB(int _r, int _g, int _b)
  {
  	r = _r;
  	g = _g;
  	b = _b;
  }
  RGB& operator=(RGB newCol)
  {
  	this->r = newCol.r;
  	this->g = newCol.g;
  	this->b = newCol.b;

  	return *this;
  }
  int r, g, b;
};

#endif
