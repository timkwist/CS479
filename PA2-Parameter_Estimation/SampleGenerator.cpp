#include "SampleGenerator.h"

vector<Vector2f> SampleGenerator::generateSamples(Vector2f mu, Matrix2f sigma)
{
	vector<Vector2f> samples;
	for(int i = 0; i < 10000; i++)
	{
		samples.push_back(Vector2f(box_muller(mu(0,0), sigma(0,0)), box_muller(mu(1,0), sigma(1,1))));
	}
	
	return samples;
}