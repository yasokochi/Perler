from joblib import Parallel, delayed
import inspect
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from scipy import optimize
from scipy.sparse import linalg
from scipy import stats
from scipy.stats import invgamma
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


#for LOOCV
def get_args_of_current_function(offset=None):
    parent_frame = inspect.currentframe().f_back
    info = inspect.getargvalues(parent_frame)
    return {key: info.locals[key] for key in info.args[offset:]}

# wrapper of PERLER method (required for joblib.Parallel and scipy.optimize)
def unwrap_self_leave_one_gene_out(arg, **kwarg):
    '''
    function for multiprocessing in loocv
    '''
    return PERLER.leave_one_gene_out(*arg, **kwarg)

def unwrap_self_k_folds_cross_validation(arg, **kwarg):
    '''
    function for multiprocessing in k_folds_cv
    '''
    return PERLER.k_folds_cross_validation(*arg, **kwarg)

def unwrap_self__F_co_mutual(A, arg):
    '''
    objective function of scipy.optimize.brute for grid search 
    '''
    return PERLER._F_co_mutual(arg, A)

def unwrap_self__F_co_mutual_list(A, arg):
    '''
    objective function of scipy.optimize.brute for grid search 
    '''
    return PERLER._F_co_mutual_list(arg, A)

def unwrap_self__F_co_mutual_cv(A, arg):
    '''
    objective function of scipy.optimize.brute for grid search 
    '''
    return PERLER._F_co_mutual_cv(arg, A)

# class implementation (_GMM and PERLER is implemented. _GMM class is inherited to PERLER class)
class _GMM():
    """
    Gausian Mixture Model (GMM) in Perler
    """
    def __init__(self, data, reference, DR = "PLSC", n_metagenes = 60, log_reference=True, zero_mean = True, print_iter=False):
        """
        initiation of GMM in Perler

        ---Paramerter---
        data: pandas.DataFrame object
        It contains scRNA-seq data (or other query data).
        Its each row represents each sample(cell) and each columns represents each genes.

        reference: pandas.DataFrame object
        It contains ISH data (or other reference data).
        Its each row represents each point(cell or region) and each columns represents each landmark genes.

        n_metagenes: int
        The number of metagenes extracted by the partial least squares correlation analysis.
        The default is 60.

        log_reference: bool
        If True, reference value are converted to log2 values. The default is True.

        ---Developers' parameter---
        Default values are recommended.

        DR: str
        The method of Dimensionality Reduction in PERLER.
        'PLSC', 'PCA', and 'NA' (No DR) are implemented
        The default is 'PLSC'.

        print_iter: bool
        If print_iter is True, likelihood is printed in each 10 steps of EM algorithm.

        zero_mean: bool 
        """
        self.__data = data
        self.__reference = reference
        self.DR = None
        self.n_metagenes = None
        self.cv = None
        self.__r_ = None
        self.__h_ = None
        self.__r = None
        self.__h = None
        self.__N = None
        self.__K = None
        self.__D = None
        self.__A = None
        self.__b = None
        self.__gamma = None
        self.__pi = None
        self.__mu = None
        self.__sigma = None
        self.__pdf = None
        self.__like = None
        self.__DM = None
        self.likelihood_logging = None


        if isinstance(print_iter, bool):
            self.print_iter = print_iter
        else:
            raise ValueError('print_iter must be Bool')

        self.metagenes_extraction(data, reference, DR, n_metagenes, log_reference, zero_mean)

    def PLSC(self, n_metagenes, zero_mean = True):
        '''
        dimensionality reduction by PLSC
        '''
        if zero_mean == True:
            X = self.__r_ - np.mean(self.__r_, axis = 0)
            Y = self.__h_ - np.mean(self.__h_, axis = 0)
        elif zero_mean == False:
            X = self.__r_
            Y = self.__h_
        else:
            raise ValueError("The parameter 'zero_mean' must be Bool.")
        mat = np.dot(X, Y.T)
        u, s, v = linalg.svds(mat, k = n_metagenes)
        return np.dot(u, np.diag(s)), np.dot(np.diag(s), v).T

    def metagenes_extraction(self, data, reference, DR, n_metagenes, log_reference=True, zero_mean = True):
        """
        metagene extraction by the partial least squares correlation analysis

        ---Paramerter---
        data: pandas.DataFrame object
        It contains scRNA-seq data (or other query data).
        Its each row represents each sample(cell) and each columns represents each genes.

        reference: pandas.DataFrame object
        It contains ISH data (or other reference data).
        Its each row represents each point(cell or region) and each columns represents each landmark genes.

        n_metagenes: int
        The number of metagenes. The default is 60.

        log_reference: bool
        If True, reference value are converted to log2 values. The default is True.
        """
        self.DR = DR
        self.n_metagenes = n_metagenes
        if log_reference:
            self.__h_ = np.log2(reference.values + 1)
        else:
            self.__h_ = reference.values
        self.__r_ = data[reference.columns].values
        if DR == 'PLSC':
            self.__r, self.__h = self.PLSC(n_metagenes, zero_mean)
        elif DR == 'PCA':
            self._pca = PCA(n_components = n_metagenes, svd_solver='full')
            self._pca.fit(self.__h_)
            self.__r = self._pca.transform(self.__r_)
            self.__h = self._pca.transform(self.__h_)
        elif DR == 'NA':
            self.__r = np.copy(self.__r_)
            self.__h = np.copy(self.__h_)
        else:
            raise ValueError("The parameter 'DR' can only accept 'PLSC', 'PCA', or 'NA'.")
        self.__N = len(self.__r) # sample size
        self.__K = len(self.__h) # reference size
        self.__D = len(self.__r[0]) # number of metagenes

    def __initiate_em(self, mu_init=None, sigma_init=None, pi_init=None, likelihood_init = -np.inf, map_sigma=False, alpha = 1.1, beta = 2):
        """
        Initiate EM algorithm

        ---Paramerter---
        Default values are recommended.

        mu_init: np.array (n_metagenes(D), reference_size(K))
        Initial mean of each Gausian distribution.

        sigma_init: np.array (n_metagenes(D))
        Initial variances of each genes.
        Perler assumes that all Gaussian distribution shares covariance matrix and that the covariances between different metagenes is 0.

        pi_init: np.array (reference_size(K))
        Inital mixing coefficients of each Gausian distribution.

        likelihood_init: int
        Initial likelihood. The default is -np.inf

        map_sigma: bool
        if True, MAP optimazation of sigma is operated. The default is False.

        alpha: float
        MAP optimazation hyper paremeter of sigma. It is used if map_sigma is Ture.
        The default is 1.1

        beta: float
        MAP optimazation hyper paremeter of sigma. It is used if map_sigma is Ture.
        The default is 2

        """
        if mu_init is None:
            mu_init = ((self.__h - np.mean(self.__h, axis=0))/np.std(self.__h, axis=0))*np.std(self.__r, axis=0) + np.mean(self.__r, axis=0)
        elif mu_init == 'default_h':
            mu_init = self.__h

        if sigma_init is None:
            s2_init = np.zeros(self.__D)
            for i in range(self.__D):
                s2_init[i] = np.var(self.__r[:,i])
            sigma_init = np.diag(s2_init)

        if pi_init is None:
            pi_init = np.ones(self.__K)/self.__K

        self.__pi = pi_init
        self.__mu = mu_init
        self.__sigma = sigma_init
        self.__calc_pdf()

        self.__like = likelihood_init

        self.map_sigma = map_sigma
        self.alpha = alpha
        self.beta = beta


    def __calc_pdf(self):
        self.__pdf = np.zeros((self.__K, self.__N))
        for j in range(self.__K):
            self.__pdf[j] = self.__pi[j]*multivariate_normal(self.__mu[j], self.__sigma).pdf(self.__r)

    def __calc_gamma(self):
        eps = 1e-300
        gamma = np.where(self.__pdf < eps, eps, self.__pdf)
        gamma /= np.sum(gamma, axis = 0)
        self.__gamma = gamma.T

    def __calc_pi(self):
        pi = np.sum(self.__gamma, axis = 0)
        pi /= np.sum(self.__gamma)
        self.__pi = pi

    def __calc_mean(self):
        H = np.sum(np.dot(self.__gamma, self.__h), axis = 0)
        X = np.sum(np.dot(self.__gamma.T, self.__r), axis = 0)
        P = np.diag(np.dot(self.__r.T, np.dot(self.__gamma, self.__h)))
        Q = np.sum(np.dot(self.__gamma, self.__h*self.__h), axis = 0)
        self.__b = (Q*X - P*H)/(self.__N*Q - H*H)
        a = (P- self.__b*H)/Q
        self.__A = np.diag(a)
        self.__mu = np.dot(self.__A, self.__h.T).T + self.__b

    def __calc_sigma(self):
        sigma = np.sum(np.dot(self.__gamma.T, self.__r * self.__r), axis = 0) \
        - 2*np.diag(np.dot(self.__r.T, np.dot(self.__gamma, self.__mu)))\
        + np.sum(np.dot(self.__gamma, self.__mu*self.__mu), axis = 0)

        if self.map_sigma:
            self.__sigma = np.diag(sigma+2*self.beta)/(self.__N+2*(self.alpha+1))

        else:
            self.__sigma = np.diag(sigma)/self.__N

    def __calc_likelihood(self):
        self.__calc_pdf()
        lh = np.sum(self.pdf, axis = 0)
        eps = 1e-300
        lh = np.where( lh < eps, eps, lh)
        logsumlh = np.sum(np.log(lh))
        if self.map_sigma:
            lh_map = np.sum(invgamma.logpdf(np.diag(self.__sigma), self.alpha, scale = self.beta))
            return logsumlh + lh_map
        else:
            return logsumlh

    def em_algorithm(self, max_n_iter = 1000, break_threshold = -0.01, print_iter = None, likelihood_logging=False, optimize_pi = True, **kwargs):
        """
        Running EM algorithm

        ---Paramerter---
        Default values are recommended.

        max_n_iter : int
        Max number of iteration of EM algorithm. The default is 1000.

        break_threshold : float
        The convergence threshold. EM iterations will stop when the lower bound average gain is over this threshold.

        print_iter : bool
        If print_iter is True, likelihood is printed in each 10 steps of EM algorithm.
        If None, self.print_iter is used.

        likelihood_logging : bool
        If likelihood_logging is True, likelihood of each step is stored into self.likelihood_logging. The default is False.

        optimize_pi : bool
        If True, mixing coefficients (pi) are optimized by EM algorithm.
        If False, mixing coefficients are fixed inital values.
        The default is True.

        **kwargs : keyword argument for self.__initiate_em

        """
        self._em_args = get_args_of_current_function(offset=1)
        for i, j in kwargs.items():
            self._em_args[i] = j

        if print_iter is None:
            print_iter = self.print_iter

        self.optimize_pi = optimize_pi

        self.__initiate_em(**kwargs)


        if likelihood_logging: self.likelihood_logging = []

        if break_threshold >= 0:
            raise ValueError('break_threshold must be a negative value.')

        for e in range(max_n_iter):
            if print_iter and e%5==0: print(e, self.__like)
            self.__calc_gamma()
            if self.optimize_pi : self.__calc_pi()
            self.__calc_mean()
            self.__calc_sigma()
            old_like = self.__like
            self.__like = self.__calc_likelihood()
            if likelihood_logging: self.likelihood_logging.append(self.__like)
            if old_like - self.__like > break_threshold: break

    def calc_dist(self):
        """
        Calculate distance between each cell and each reference datapoint
        """
        VI = np.linalg.inv(self.__sigma)
        DM = distance.cdist(self.__mu, self.__r, V=VI, metric="mahalanobis")
        self.__DM = DM

    def _test(self, a = 1, b = 2):
        c = 'c'
        self._test = get_args_of_current_function(offset=1)
        return get_args_of_current_function(offset=1)


    # properties and setters (to access parameters)
    @property
    def data(self):
        return self.__data

    @property
    def ref(self):
        return self.__reference

    @property
    def r(self):
        return np.copy(self.__r)

    @property
    def h(self):
        return np.copy(self.__h)

    @property    
    def r_(self):
        return np.copy(self.__r_)

    @property
    def h_(self):
        return np.copy(self.__h_)

    @property
    def N(self):
        return self.__N

    @property
    def K(self):
        return self.__K

    @property
    def D(self):
        return self.__D

    @property
    def pdf(self):
        return np.copy(self.__pdf)

    @property
    def gamma(self):
        return np.copy(self.__gamma)

    @gamma.setter
    def gamma(self, value):
        self.__gamma = value

    @property
    def pi(self):
        return np.copy(self.__pi)

    @pi.setter
    def pi(self, value):
        self.__pi = value

    @property
    def sigma(self):
        return np.copy(self.__sigma)

    @property
    def mu(self):
        return np.copy(self.__mu)

    @property
    def A(self):
        return np.copy(self.__A)

    @property
    def b(self):
        return np.copy(self.__b)

    @property
    def likelihood(self):
        return self.__like

    @property
    def DM(self):
        return self.__DM

    @DM.setter
    def DM(self, value):
        self.__DM = value


class PERLER(_GMM):
    """
    PERLER object
    """
    def __init__(self, data, reference, DR = "PLSC", n_metagenes = 60, log_reference=True, zero_mean = True, print_iter=False):
        super().__init__(data, reference, DR, n_metagenes, log_reference, zero_mean, print_iter)
        self.__loocv = None
        self.__M = None
        self.__P = None
        self.__res = None
        self.__result = None
        self.__location = None
        self.__result_with_location = pd.DataFrame()


    def leave_one_gene_out(self, gene):
        '''
        wrapper of loo function
        '''
        print(gene)
        reference = self.ref.drop(gene, axis=1)
        gmm = _GMM(data = self.data, reference= reference, DR = self.DR, n_metagenes = self.n_metagenes, print_iter=True)
        gmm.em_algorithm(**self._em_args)
        gmm.calc_dist()
        return gene, gmm.pi, gmm.DM

    def k_folds_cross_validation(self, genes, cycle):
        '''
        wrapper of loo function
        '''
        print(cycle)
        reference = self.ref.drop(columns = genes, axis=1)
        gmm = _GMM(data = self.data, reference= reference, DR = self.DR, n_metagenes = self.n_metagenes, print_iter=True)
        gmm.em_algorithm(**self._em_args)
        gmm.calc_dist()
        return cycle, gmm.pi, gmm.DM

    def loocv(self, workers = -1):
        '''
        running loocv experiments

        ---parameters---
        Default values are recommended.

        workers : int
        numbers of workers in multiprocessing using joblib. 
        The default is -1 (using the max numbers of workers in your computer)
        '''
        loo_results = Parallel(n_jobs = workers, verbose = 10)( [delayed(unwrap_self_leave_one_gene_out)(args) for args in zip([self]*len(self.ref.columns), self.ref.columns)] )
        #sorting
        ##making a dictionary
        loo_results_dict = {}
        for items in loo_results:
            loo_results_dict[items[0]] = [items[1], items[2]]
        self.__loocv = []
        for gene in self.ref.columns:
            self.__loocv.append(loo_results_dict[gene])

    def k_folds_cv(self, k = 10, seed = 10, workers = -1):
        '''
        running loocv experiments

        ---parameters---
        Default values are recommended.

        k : int 

        seed : int

        workers : int
        numbers of workers in multiprocessing using joblib. 
        The default is -1 (using the max numbers of workers in your computer)
        '''
        np.random.seed(seed)
        self.cv = True
        num_gene = len(self.ref.columns)
        self._id_all   = np.random.choice(num_gene, num_gene, replace=False)
        self._cv_k = []
        self._k = k
        for i in range(k):
            self._cv_k.append(self.ref.columns[self._id_all[i*(num_gene//self._k + 1): (i+1)*(num_gene//self._k + 1)]])
        cv_results = Parallel(n_jobs = workers, verbose = 10)( [delayed(unwrap_self_k_folds_cross_validation)(args) for args in zip([self]*len(self._cv_k), self._cv_k, range(len(self._cv_k)))] )
        #sorting
        ##making a dictionary
        cv_results_dict = {}
        for items in cv_results:
            cv_results_dict[items[0]] = [items[1], items[2]]
        self.__fold_cv = []
        for i in range(self._k):
            self.__fold_cv.append(cv_results_dict[i])    

    def __calc_M_and_P(self):
        '''
        change the data structure for the hyperparameters estimation
        '''
        self.__M = np.zeros((self.ref.shape[1], self.K, self.data.shape[0]))
        self.__P = np.zeros((self.ref.shape[1], self.K))
        for i, result in enumerate(self.__loocv):
            self.__P[i] = result[0]
            self.__M[i] = result[1]

    def _F_co_mutual(self, A):
        '''
        the objective function of the hyperparameters estimation
        '''
        Mu = np.zeros([self.ref.shape[1], self.K])
        for i in range(self.ref.shape[1]):
            w_i = (self.__P[i]*np.exp(-(A[0]**2)*(self.__M[i]**2)-(A[1]**2)*self.__M[i]).T).T
            w_i = w_i/ np.sum(w_i, axis = 0)
            Mu[i] = (np.dot(w_i, self.r_[:,i]).T/np.sum(w_i, axis=1)).T
        co_mutual = np.trace(np.log(1-np.corrcoef(self.h_.T, Mu)[self.ref.shape[1]:,:self.ref.shape[1]]**2))
        return co_mutual

    
    def _rec_i(self, i, A):
        w_i = (self.__loocv[i][0]*np.exp(-(A[0]**2)*(self.__loocv[i][1]**2)-(A[1]**2)*self.__loocv[i][1]).T).T
        w_i = w_i/ np.sum(w_i, axis = 0)
        return (np.dot(w_i, self.r_[:,i]).T/np.sum(w_i, axis=1)).T

    def _rec_cv_i(self, i, A):
        num_gene = len(self.ref.columns)
        w_i = (self.__fold_cv[i][0]*np.exp(-(A[0]**2)*(self.__fold_cv[i][1]**2)-(A[1]**2)*self.__fold_cv[i][1]).T).T
        w_i = w_i/ np.sum(w_i, axis = 0)
        return (np.dot(w_i, self.r_[:,self._id_all[i*(num_gene//self._k + 1):(i+1)*(num_gene//self._k + 1)]]).T/np.sum(w_i, axis=1)).T


    def _F_co_mutual_list(self, A):
        '''
        the objective function of the hyperparameters estimation
        '''
        Mu = np.stack([self._rec_i(i, A) for i in range(len(self.ref.columns))])
        co_mutual = np.trace(np.log(1-np.corrcoef(self.h_.T, Mu)[self.ref.shape[1]:,:self.ref.shape[1]]**2))
        return co_mutual

    
    def _F_co_mutual_cv(self, A):
        '''
        the objective function of the hyperparameters estimation
        '''
        num_gene = len(self.ref.columns)
        co = np.zeros(self._k)
        for l in range(self._k):
            Mu = self._rec_cv_i(l, A)
            co[l] = np.trace(np.log(1 - np.corrcoef(self.h_[:, self._id_all[l*(num_gene//self._k + 1):(l+1)*(num_gene//self._k + 1)]].T, Mu.T)[:Mu.shape[1], Mu.shape[1]:]**2))
        co_mutual = np.sum(co)
        return co_mutual



    
    def grid_search(self, grids = ((0,1), (0,1)), workers = -1):
        '''
        grid search for the hyperparameters estimation by using scipy.optimize.brute

        ---parameters---
        grids : tuple
        set the ranges parameters of scipy.optimize.brute function.
        The default is ((0,1), (0,1)).

        ---Developers' parameter---
        Default values are recommended.

        workers : int
        numbers of workers in multiprocessing. 
        The default is -1 (using the max numbers of workers in your computer)
        '''
        if self.cv is None:
            self.__res = optimize.brute(unwrap_self__F_co_mutual_list, ranges = grids, args = (self,), workers = workers)
        elif self.cv == True:
            self.__res = optimize.brute(unwrap_self__F_co_mutual_cv, ranges = grids, args = (self,), workers = workers)
        else:
            raise ValueError("The parameter 'plr.cv' must be Bool.")

    def spatial_reconstruction(self, z_scored = True, location = None, mirror = False, _3d = False, _res = True):
        '''
        function for spatial reconstruction (the second step of perler)

        ---parameters---
        location : pandas.DataFrame object, optional
        If you have cell location data of ISH data, you can add location data to the result of perler through this parameter.
        This pandas.DataFrame object must have columns which specify x_axis and y_axis (and z_axis for 3_dimensional data) of the coordinates of the cells.
        the default is None

        mirror : bool, only requierd in the Dmel dataset (Karaiskos., et al, 2017)
        In Dmel dataset, the result of perler must be mirrored for visualization. Please see Methods in our manuscripts and Karaiskos, et al., 2017
        The default is False

        _3d : bool, only requierd in the Dmel dataset (Karaiskos., et al, 2017)
        In Dmel dataset, the columns of cell location dataframe is changed from ['x_coord'...] to ['X'...] in our implementation for the clarity of the code. 
        The default is False

        ---Developers' parameter---
        Default values are recommended.

        _res : bool
        If False, res**2 is set to [-0.5, 0] (the unoptimized hyperparameter value) 
        '''
        if isinstance(_res, list):
            weight = (self.pi*np.exp(-(_res[0]**2)*(self.DM**2)-(_res[1]**2)*self.DM).T).T
            weight = weight/ np.sum(weight, axis = 0)
        elif _res == True:
            weight = (self.pi*np.exp(-(self.__res[0]**2)*(self.DM**2)-(self.__res[1]**2)*self.DM).T).T
            weight = weight/ np.sum(weight, axis = 0)
        elif _res == False:
            weight = self.gamma.T
        q = (np.dot(weight, self.data.values).T/np.sum(weight, axis=1)).T
        if z_scored == True :
            e=stats.zscore(q, axis=0)
            Result = pd.DataFrame(e, columns=self.data.columns)
        elif z_scored == False:
            Result = pd.DataFrame(q, columns=self.data.columns)
        else:
            raise ValueError("the parameter 'z_scored' must be Bool")

        if mirror == True:
            comp_norm = pd.concat([Result, Result], axis=0).reset_index(drop=True)
            self.__result = comp_norm
        elif mirror == False:
            self.__result = Result
        else:
            raise ValueError("the parameter 'mirror' must be Bool")
        if not location is None:
            self.__location = location
            if _3d == True:
                self.__location.columns = ["X", "Y", "Z"]
            self.__result_with_location = pd.concat([self.__location, self.__result], axis=1)

    def Dmel_visualization(self, gene, view = 'lateral', color_map = 'BuPu'):
        '''
        function for visualization of Dmel datasets

        --parameters---
        gene : str
        gene name you want to visualize

        view : str
        set the view of the Dmel virtual embryo. This parameter must be among "lateral", "anterior", "posterior", "top", and "bottom".
        The default is "lateral"

        color_map : str
        color map of plt.scatter() function
        The default is "BuPu"
        '''
        if self.__result_with_location.empty:
            raise ValueError("There is no location data")
        else:
            if view == "lateral":
                fig = plt.figure()
                plt.scatter(self.__result_with_location["X"], self.__result_with_location["Z"], c= self.__result_with_location[gene], cmap = color_map)
                plt.title(gene)
                plt.colorbar()
                fig.show()
            elif view == "top":
                fig = plt.figure()
                plt.scatter(self.__result_with_location[self.__result_with_location["Z"]>0]["X"],self.__result_with_location[self.__result_with_location["Z"]>0]["Y"], c=self.__result_with_location[self.__result_with_location["Z"]>0][gene], cmap = color_map)
                plt.title(gene)
                plt.colorbar()
                fig.show()
            elif view == 'bottom':
                fig = plt.figure()
                plt.scatter(self.__result_with_location[self.__result_with_location["Z"]<0]["X"],self.__result_with_location[self.__result_with_location["Z"]<0]["Y"], c=self.__result_with_location[self.__result_with_location["Z"]<0][gene], cmap = color_map)
                plt.title(gene)
                plt.colorbar()
                fig.show()
            elif view == 'anterior':
                fig = plt.figure()
                plt.scatter(self.__result_with_location[self.__result_with_location["X"]<0]["Y"],self.__result_with_location[self.__result_with_location["X"]<0]["Z"], c=self.__result_with_location[self.__result_with_location["X"]<0][gene], cmap = color_map)
                plt.title(gene)
                plt.colorbar()
                fig.show()
            elif view == 'posterior':
                fig = plt.figure()
                plt.scatter(self.__result_with_location[self.__result_with_location["X"]>0]["Y"],self.__result_with_location[self.__result_with_location["X"]>0]["Z"], c=self.__result_with_location[self.__result_with_location["X"]>0][gene], cmap = color_map)
                plt.title(gene)
                plt.colorbar()
                fig.show()

            else:
                raise ValueError("The parameter 'view' cannot accept this value. Please see the description of this function")



    # properties and setters (to access parameters)
    @property
    def loocv_result(self):
        return self.__loocv

    @loocv_result.setter
    def loocv_result(self, value):
        self.__loocv = value

    @property
    def res(self):
        return np.copy(self.__res)
    
    @res.setter
    def res(self, value):
        self.__res = value

    @property
    def location(self):
        return self.__location

    @property
    def result(self):
        return self.__result

    @property
    def result_with_location(self):
        return self.__result_with_location
