# geodesicLVQ_toolbox

This code demonstrates how Matrix Learning Vector Quantization models can be averaged to achieve an interpretable ensemble. Parts can also be used to find rank preserving averages of positive semi-definite matrices, as used in Large Margin Nearest Neighbors and others with adaptive quadratic forms. 

The implementation of the rank preserving average of semi-definite matrices is based on code and the paper of Silvère Bonnabel, Anne Collard and Rodolphe Sepulchre from 2013:  
"Rank-preserving geometric means of positive semi-definite matrices", Linear Algebra and its Applications. Vol. 438(8), pp. 3202 - 3216. 
https://doi.org/10.1016/j.laa.2012.12.009

## Before you can start

This demo uses implementations of manifold distance computations and, as well as Karcher and Ando mean from a couple of manifold learning toolboxes as listed below. Please download them and add them to the Matlab path before trying to run the script.

### angle LVQ toolbox

To run this demo also download the angleLVQtoolbox available at:  
https://github.com/kbunte/angleLVQtoolbox.git  
and add it to the Matlab path before running the demo script.

### Manopt

Furthermore the Manopt, the Matlab toolbox for optimization on manifolds needs to be downloaded and added to the matlab path. 
For a description of the project, documentation, examples and more, see:
[http://www.manopt.org](http://www.manopt.org).


Manopt is released in numbered versions from time to time. For most users, it is easiest to download the latest numbered version from  
[http://www.manopt.org/downloads.html](http://www.manopt.org/downloads.html).

### Subspace Mean and Median Evaluation Toolkit (SuMMET)

SuMMET is a Matlab-based software package that is meant as a companion for the paper:  
"Finding the Subspace Mean or Median to Fit Your Need," that has been accepted for the 2014 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2014).

The author of this software package is Tim Marrinan and source code can be downloaded from:  
https://www.cs.colostate.edu/~vision/summet/

### Matrix Means Toolbox (mmtoolbox), from Bini 2010

The toolbox was available to download at: http://bezout.dm.unipi.it/software/mmtoolbox/  
Last time I checked it was not responsive anymore and I added the required file under *tools*.  
