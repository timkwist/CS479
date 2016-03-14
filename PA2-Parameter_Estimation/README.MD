CS 479/679 Pattern Recognition
Spring 2016 – Prof. Bebis
Programming Assignment 2 - Due: 3/14/2016

Team: Shane Melton, Tim Kwist

Estimating Parameters Using Maximum Likelihood Estimation

Tested on two sets of data:

Set 1:

muOne = [1; 1]
muTwo = [6; 6]
sigmaOne = sigmaTwo = [2 0; 0 2]

Set 2:
muOne = [1; 1]
muTwo = [6; 6]
sigmaOne = [2 0; 0 2]
sigmaTwo = [4 0; 0 8]

Part One (A): Using 10,000 samples from each distribution in Set 1, estimate parameters using Maximum Likelihood Estimation and re-classify based on estimate parameters

Part One (B): Using 1,000 samples from each distribution in Set 1, estimate parameters using Maximum Likelihood Estimation and re-classify based on estimate parameters

Part Two: Repeat Part One (A and B) using samples from each distribution in Set 2.

Part Three (A): Implement skin-color methodology outlined in Section 3.1 of [Yang96 "A Real-time Face Tracker"].

Part Three (B): Repeat Three (A) using the YCbCr color space.

Procedure to run program:

Part 1 and 2:

make MLE-1-2.exe
./MLE-1-2.exe

Part 3:

make MLE-3.exe
./MLE-3.exe

Results for part 1 and 2 are in 'results' folder

PA1-Output.txt: Displays statistics regarding number of samples misclassified for each part of program

Part-*-Misclassified.txt: List of all points misclassified for that part, in the format of sampleOne[x] sampleOne[y] sampleTwo[x] sampleTwo[y]

Part-*.txt: List of all samples for given part, in the format of sampleOne[x] sampleOne[y] sampleTwo[x] sampleTwo[y]

Results for part 3 are in the following files:

Train#-RGB-ROC-Data.txt: Shows the False Negative and False Positive rates for each threshold using the RGB color space
Train#-YCC-ROC-Data.txt: Shows the False Negative and False Positive rates for each threshold using the YCC color space
Train#-RGB-2Class.txt: Shows the False Negative and False Positive rates for the 2 class Bayesian Classifier using the RGB color space
Train#-YCC-2Class.txt: Shows the False Negative and False Positive rates for the 2 class Bayesian Classifier using the YCC color space