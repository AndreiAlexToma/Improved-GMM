# Improved-GMM
Development of a Very Low-Cost Deforestation Monitoring System Based on Aerial Image Clustering and Compression Techniques

These Python scripts implement a cost-reduction approach to GMM clustering using three discrete transforms usually used in compression problems.

The repository is structured to follow the published article.

The file names starting with "Batch" are the iterative implementation for the entire dataset.

GMM.py and Batch_GMM.py are the GMM clustering implementations, without any improvements, which served as a baseline for the proposed methods.

Several files can be found in the DCT, DFT & DWT folders. The "GMM_XXX" and "Batch_GMM_XXX" files contain the GMM clustering augmented with the respective discrete transform and the other workflow steps described in the paper. The other files named "XXX_1BAND" or "XXX_4BAND" transform a multispectral image to the frequency domain, without any clustering attached. 

The Evaluations folder is used for calculating the DBI score of the results.
