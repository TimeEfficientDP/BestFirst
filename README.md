# BestFirst

Implementation of the algorithms:
 * Mint (ViterbiOptimalPath) for efficient decoding; 
 * Tech (HistogramConstruction) for efficient histogram construction. 

The algorithms are based on the ISABELLA framework. 

Scripts and datasets are also provided. 

More in detail, the structure of the repository is as follows: 

- HistogramConstruction: 
	- BestFirstHistogram.py: different algorithm for histogram construction. 
 	- dataHistogram : containing data used for the expriments with histogram construction (DowJones).
 	- Histogram\_main\_DowJones\_data : script to run histogram construction experiments with real data (by setting appropriate parameters).
	- Histogram\_main\_synthetic\_data : script to run histogram construction experiments with synthetic data (by setting appropriate parameters).



- ViterbiOptimalPath
	- bestFirstViterbi.py: different algorithm for Viterbi optimal path. 
	- BestFirstViterbi_Decoding.py: : different algorithm for Viterbi optimal path for decoding data.
	- dataOptimalPath:  : containing data used for the expriments with Viterbi-based decoding.
	- Main\_ViterbiOptimalPath\_DecodingData.py: script to run decoding experiments with real data (by setting appropriate parameters).
	- Main\_ViterbiOptimalPath\_SyntheticData.py : script to run optimal Viterbi path experiments with synthetic data (by setting appropriate parameters).
	- graph.py: class used to represent transition graphs and their properties.
		

In addition we provide a simple script performing a basic test, checking the optimality of Mint.  

Our implementation uses the PQDict module (see: https://pypi.org/project/pqdict/). 

Note that the data used in the forced alignment shared cannot be shared in this repository due to their size. 
However, they can be provided upon request. 
For such data and related code as well as for further information, please contact martino.ciaperoni@aalto.fi.
