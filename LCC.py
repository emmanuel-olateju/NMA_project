import copy
import numpy as np
import random

class LCC_param:
    def __init__(self,Xs):
        try:
            for x in Xs:
                assert x.ndim==2 and x.shape==Xs[0].shape
        except AssertionError:
            raise AssertionError('Ensure X is an iterable of 2D arrays of the same shape')
        try:
            assert len(Xs) <= Xs[0].shape[1]
        except AssertionError:
            raise AssertionError('Number of multi-dimension variable should not be greater than dimension of each variable')
        
        self.Xs=Xs
        self._m=len(self.Xs)
        self.__n=Xs[0].shape[1]
        self.__l=Xs[0].shape[0]

        self.Xcorr=0
        self.Ycorr=0

        self.Xs=[(np.nanmax(j)-j)/(np.nanmax(j)-np.nanmin(j)) for j in self.Xs]

        self.__stdzd_vector()
        self._inner_product_matrix()
        self.XY_corr()
        self.Xcorr=self._within_corr(0)
        self.Ycorr=self._within_corr(1)
        self._multiple_corr_coef()
        self._linearity=np.sum(np.abs(np.diag(self.XYcorr)))/len(self.XYcorr)

    def __stdzd_vector(self):
        self.__stdzd_vectors=np.empty((0,self.__n,self.__l))
        for x in self.Xs:
            mean=np.nanmean(x,0)
            std=np.nanstd(x,0)
            vectors=list()
            for c in range(self.__n):
                # vectors.append(x[:,c])
                vectors.append((x[:,c]-mean[c])/std[c])
            self.__stdzd_vectors=np.vstack((self.__stdzd_vectors,np.array([vectors])))
    
    def _inner_product_matrix(self):
        self.ipm=np.empty((0,self.__n))
        for c1 in range (self.__n):
            row=list()
            for c2 in range(self.__n):
                # row.append(np.corrcoef(self.__stdzd_vectors[0,c1,:],self.__stdzd_vectors[1,c2,:])[0,1])
                row.append(np.inner(self.__stdzd_vectors[0,c1,:],self.__stdzd_vectors[1,c2,:]))
            row=np.array(row)
            row=np.nan_to_num(row,True,0,np.nanmean(row),np.nanmean(row))
            self.ipm=np.vstack((self.ipm,np.array(row)))  
        # self.ipm=(self.ipm-np.nanmean(self.ipm))/np.nanstd(self.ipm)
        self.ipm=(self.ipm-np.nanmin(self.ipm))/(np.nanmax(self.ipm)-np.nanmin(self.ipm))
        # self.ipm=np.nan_to_num(self.ipm,True,0,np.nan,self.ipm.min())

    def XY_corr(self):
        x=self.Xs[0]
        y=self.Xs[1]
        self.XYcorr=np.empty((0,x.shape[1]))
        for c1 in range(x.shape[1]):
            row=list()
            for c2 in range(x.shape[1]):
                row.append(np.corrcoef(x[:,c1],y[:,c2])[0,1])
            row=np.array(row)
            row=np.nan_to_num(row,True,0,np.nanmean(row),np.nanmean(row))
            self.XYcorr=np.vstack((self.XYcorr,row))

    def _within_corr(self,i):
        acorr=np.empty((0,self.__n))
        for c1 in range(self.__n):
            row=list()
            for c2 in range(self.__n):
                row.append(np.inner(self.Xs[i][c1,:],self.Xs[i][c2,:]))
            row=np.array(row)
            row=np.nan_to_num(row,True,0,np.nanmean(row),np.nanmean(row))
            acorr=np.vstack((acorr,row))
        return acorr


    def _multiple_corr_coef(self):
        S=list()
        for c1 in range(self.__n):
            s=np.sqrt(self.XYcorr[:,c1] @ self.Xcorr @ self.XYcorr[:,c1].T)
            S.append(s)
        
        self.m_corr_coef=np.array(S)