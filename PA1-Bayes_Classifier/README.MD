CS 479/679 Pattern Recognition
Spring 2016 – Prof. Bebis
Programming Assignment 1 - Due: 2/22/2016

Team: Shane Melton, Tim Kwist

2D Gaussian Distribution sample classifier

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

Also calculates Chernoff Bound and Bhattacharyya Bound

Minimum-Distance Classifier used on Set 2 to demonstrate misuse of Minimum-Distance Classifier

Part One (A): Classify Set 1 samples, with priors = 0.5 for both distributions

Part One (B): Classify Set 1 samples, with Prior One = 0.2 and Prior Two = 0.8

Part One (A): Classify Set 2 samples, with priors = 0.5 for both distributions

Part One (B): Classify Set 2 samples, with Prior One = 0.2 and Prior Two = 0.8

Part Three: Classify Set 2 samples with Minimum-Distance Classifier

Procedure to run program:

make
./BayesClassifier.exe

Results are in 'results' folder

PA1-Output.txt: Displays statistics regarding number of samples misclassified for each part of program; Also gives Chernoff and Bhattacharyya Bound information

Part-*-Misclassified.txt: List of all points misclassified for that part, in the format of sampleOne[x] sampleOne[y] sampleTwo[x] sampleTwo[y]

Part-*.txt: List of all samples for given part, in the format of sampleOne[x] sampleOne[y] sampleTwo[x] sampleTwo[y]