#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

using namespace Eigen;
using namespace std;

extern float box_muller(float, float);

class BayesClassifier
{
public:	
	/**
	 * Calculate the discriminant for muOne/varianceOne, muTwo/varianceTwo against input x
	 * If priors are not the same, add log of prior to each discriminant.
	 * Classifies sample by whichever discriminant is larger.
	 * 
	 * Case I classification only possible if each Sigma is equal to a variance * identity matrix
	 * 
	 * @param x 2x1 input sample
	 * @param muOne 2x1 mu vector
	 * @param muTwo 2x1 mu vector
	 * @param varianceOne Scalar variance
	 * @param varianceTwo Scalar variance
	 * @param priorOne Optional; 0 <= priorOne <= 1; 1 - priorOne = priorTwo
	 * @param priorTwo Optiona; 0 <= priorTwo <= 1; 1 - priorTwo = priorOne
	 * @return 1 if discriminant of parameters for one > discriminant of parameters for two; Two otherwise
	 */
	static int classifierCaseOne(Vector2f x, Vector2f muOne, Vector2f muTwo, float varianceOne, float varianceTwo, float priorOne = 0.5, float priorTwo = 0.5);
	
	/**
	 * Calculate the discriminant for muOne/varianceOne, muTwo/varianceTwo against input x
	 * If priors are not the same, add log of prior to each discriminant.
	 * Classifies sample by whichever discriminant is larger.
	 * 
	 * Case I classification only possible if each Sigma is equal to the same Sigma
	 * 
	 * @param x 2x1 input sample
	 * @param muOne 2x1 mu vector
	 * @param muTwo 2x1 mu vector
	 * @param varianceOne Scalar variance
		 * @param varianceTwo Scalar variance
	 * @param priorOne Optional; 0 <= priorOne <= 1; 1 - priorOne = priorTwo
	 * @param priorTwo Optiona; 0 <= priorTwo <= 1; 1 - priorTwo = priorOne
	 * @return 1 if discriminant of parameters for one > discriminant of parameters for two; Two otherwise
	 */
	static int classifierCaseTwo(Vector2f x, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo, float priorOne = 0.5, float priorTwo = 0.5);
	
	/**
	 * Calculate the discriminant for muOne/varianceOne, muTwo/varianceTwo against input x
	 * If priors are not the same, add log of prior to each discriminant.
	 * Classifies sample by whichever discriminant is larger.
	 * 
	 * Case I classification only possible if each Sigma is equal to some arbitrary Sigma
	 * 
	 * @param x 2x1 input sample
	 * @param muOne 2x1 mu vector
	 * @param muTwo 2x1 mu vector
	 * @param varianceOne Scalar variance
	 * @param varianceTwo Scalar variance
	 * @param priorOne Optional; 0 <= priorOne <= 1; 1 - priorOne = priorTwo
	 * @param priorTwo Optiona; 0 <= priorTwo <= 1; 1 - priorTwo = priorOne
	 * @return 1 if discriminant of parameters for one > discriminant of parameters for two; Two otherwise
	 */
	static int classifierCaseThree(Vector2f x, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo, float priorOne = 0.5, float priorTwo = 0.5);
		
	/**
	 * Calculate the discriminant for muOne/varianceOne, muTwo/varianceTwo against input x
	 * If priors are not the same, add log of prior to each discriminant.
	 * Classifies sample by whichever discriminant is larger.
	 * 
	 * Case I classification only possible if each Sigma is equal to some arbitrary Sigma
	 * 
	 * @param x 2x1 input sample
	 * @param muOne 2x1 mu vector
	 * @param muTwo 2x1 mu vector
	 * @param varianceOne Scalar variance
	 * @param varianceTwo Scalar variance
	 * @param priorOne Optional; 0 <= priorOne <= 1; 1 - priorOne = priorTwo
	 * @param priorTwo Optiona; 0 <= priorTwo <= 1; 1 - priorTwo = priorOne
	 * @return 1 if discriminant of parameters for one > discriminant of parameters for two; Two otherwise
	 */
	static int classifierCaseThree(VectorXf x, VectorXf muOne, VectorXf muTwo, MatrixXf sigmaOne, MatrixXf sigmaTwo, float priorOne = 0.5, float priorTwo = 0.5);

	/**
	 * Determine whether a given 2x1 sample is a member of the Gaussian Distribution
	 * Specified by the given mu and sigma by calculating the PDF and comparing if it is
	 * greater than the given threshold.
	 *
	 * @param x 2x1 sample data
	 * @param mu 2x1 mean of the Gaussian Distribution
	 * @param sigma 2x2 covariance of the Gaussian Distribution
	 * @param threshold Some float number to be used as a threshold
	 * @return Returns true if PDF(x) ~ Gaussian(mu, sigma) > threshold; false otherwise
	 */
	static bool thresholdCaseThree(Vector2f x, Vector2f mu, Matrix2f sigma, float threshold);

	/**
	 * Calculates the discriminants for minimum distance classifier
	 * 
	 * Minimum distance classifier only valid if priors of each distribution are equal and the
	 * distributions belong to Case I
	 * 
	 * @param x 2x1 input sample
	 * @param muOne 2x1 mu vector
	 * @param muTwo 2x1 mu vector
	 * @return 1 if discriminant of parameters for one > discriminant of parameters for two; Two otherwise
	 */
	static int minimumDistanceClassifier(Vector2f x, Vector2f muOne, Vector2f muTwo);
	
	/**
	 * Locates the Chernoff Bound by minimizing e^-k(b)
	 * Accurate to +/- 0.00001 of the index
	 * 
	 * @param muOne 2x1 mu vector
	 * @param muTwo 2x1 mu vector
	 * @param sigmaOne 2x2 sigma matrix
	 * @param sigmaTwo 2x2 sigma matrix
	 * @return Pair consisting of (ChernoffIndex, e^-k(ChernoffIndex))
	 */
	static pair<float, float> findChernoffBound(Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo);
	
	/**
	 * Returns e^-(k(0.5))
	 * 
	 * @param muOne 2x1 mu vector
	 * @param muTwo 2x1 mu vector
	 * @param sigmaOne 2x2 sigma matrix
	 * @param sigmaTwo 2x2 sigma matrix
	 * @return e^-(k(0.5))
	 */
	static float findBhattacharyyaBound(Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo);
private:
	/**
	 * Returns norm squared of given vector
	 * 
	 * Norm squared of x defined as transpose(x) * x
	 * 
	 * @param x 2x1 input vector
	 * @return transpose(x) * x
	 */
	static float normSquared(Vector2f x);

	/**
	 * Returns e^-k(beta) for given beta
	 * 
	 * @param beta 0 <= beta <= 1
	 * @param muOne 2x1 mu vector
	 * @param muTwo 2x1 mu vector
	 * @param sigmaOne 2x2 sigma matrix
	 * @param sigmaTwo 2x2 sigma matrix
	 * @return e^-k(beta)
	 */
	static float errorBound(float beta, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo);
};

#endif
