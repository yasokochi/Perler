# PERLER: Probabilistic Embyo Reconstruction by Linear Evaluation of scRNAseq

Perler is a novel model-based computational method which reconstructs spatial gene-expression profiles from scRNAseq data via generative linear modeling in a biologically interpretable framework.
***
## Getting Started
### How to install
##### Dependencies and requirements
Perler is built upon the following packages...
- joblib, numpy, pandas, scipy, scikit-learn, and matplotlib

If these modules do not exist, they are automatically installed.
##### installation
You can install Perler through [pip](https://pypi.org/project/pip/) command. Installation normally takes less than 1min. <br> 
pip
```python
pip install git+https://github.com/yasokochi/Perler.git
```
pipenv
```python
pipenv install git+https://github.com/yasokochi/Perler.git#egg=perler
```

### environment
The environment where Perler has been developed is below...

Mac OS
```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.15.7
    BuildVersion:	19H2


Python
```python
import sys
sys.version
```




    '3.8.3 (default, Jul 14 2020, 15:24:14) \n[Clang 11.0.3 (clang-1103.0.32.62)]'
also, the version information of the required modules is in [requirements.txt](requirements.txt)
***
## Usage
### PERLER procedures
This is a very short introduction of Perler.<br>
As input, Perler take

- scRNAseq data with sample rows and gene columns
- *in situ* data (ISH) with sample rows and gene columns
- (*optional*) location data of each *in situ* data points


```python
import perler
#Making PERLER object
plr = perler.PERLER(data = scRNAseq, reference=ISH)
#Generative linear mapping (the first step of perler)
##The parameter fitting by EM algorithm
plr.em_algorithm(optimize_pi = False)
##Calculate the pair-wise distance between scRNAseq data and reference data
plr.calc_dist()
#Hyperparameter estimation
##conducting LOOCV experiment
plr.loocv()
##fitting the hyperparameters by grid search
plr.grid_search()
#spatial reconstruction (the second step of perler)
plr.spatial_reconstruction(location = location)
#show results
print(plr.result_with_location.head())
```
For more information, please see [examples](examples).

If you find this work is useful, please cite: Yasushi Okochi, Shunta Sakaguchi, Ken Nakae, Takefumi Kondo, and Honda Naoki. "Model-based prediction of spatial gene expression via generative linear mapping", bioRxiv (2020) [URL](https://www.biorxiv.org/content/10.1101/2020.05.21.107847v1.full)

The all source codes used in this study are deposited at [Google Drive](https://drive.google.com/file/d/1sCX4AcyoVl-8P2QPS0wz6xLobENsIao4/view?usp=sharing).