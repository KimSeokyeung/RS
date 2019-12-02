import numpy as np
import pandas as pd

class Basic_CF_Algorithm:
    def __init__(self, pre_MovieLens):
        self.pre_MovieLens=np.array(pre_MovieLens)

    def basic_baseline_user(self,sim, k):
        self.sim=sim
        self.k=k
        
        #예측값 넣을 빈 행렬 생성
        predicted_rating = np.array([[0.0 for col in range(500)] for row in range(500)])
    
        #평균 & 표준편차 - 0값은 NaN값으로 변환
        mean_u = np.nanmean(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan), axis = 1) #user
        mean_i = np.nanmean(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan), axis = 0) #item
        mean = np.nanmean(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan)) #all mean
    
        #baseline
        b_u = mean_u - mean #user
        b_i = mean_i - mean #item
    
        #Similarty 불러오기
        Sim=sim
    
        #상위 K명 구하기
        k_neighbors = np.argsort(-Sim) 
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1) #k개만 남기고 삭제
    
        NumUsers = np.size(self.pre_MovieLens, axis = 0) #user size
        NumItems = np.size(self.pre_MovieLens, axis = 1) #item size
    
        for u in range(0, NumUsers):
            list_sim = Sim[u, k_neighbors[u]] #similarity list
            for i in range(0, NumItems):
                list_rating = self.pre_MovieLens[k_neighbors[u],i].astype('float64') #rating list
            
                b_ui = mean + b_u[u] + b_i[i] #user u, item i
                b_vi = mean + b_u[k_neighbors[u]] + b_i[i]
        
                #예측값 계산
                denominator = np.sum(list_sim) #분모
                numerator = np.sum(list_sim * (list_rating - b_vi)) #분자
                predicted_rating[u,i] = b_ui + numerator / denominator #예측값 계산 완료
        
        return predicted_rating


    def basic_baseline_item(self,sim, k):
        self.sim=sim
        self.k=k
        
        #예측값 넣을 빈 행렬 생성
        predicted_rating = np.array([[0.0 for col in range(500)] for row in range(500)])
        
        #평균 & 표준편차 - 0값은 NaN값으로 변환
        mean_u = np.nanmean(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan), axis = 1) #user
        mean_i = np.nanmean(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan), axis = 0) #item
        mean = np.nanmean(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan)) #all
        
        #baseline
        b_u = mean_u - mean #user
        b_i = mean_i - mean #item
            
        self.pre_MovieLens = (self.pre_MovieLens).T
        
        #Similarty 불러오기
        Sim=sim
        
        self.pre_MovieLens = (self.pre_MovieLens).T
        
        #상위 K명 구하기
        k_neighbors = np.argsort(-Sim) 
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1) #k개만 남기고 삭제
        
        NumUsers = np.size(self.pre_MovieLens, axis = 0) #user size
        NumItems = np.size(self.pre_MovieLens, axis = 1) #tiem size
        
        for i in range(0, NumItems):
            list_sim = Sim[i, k_neighbors[i]] #similarity list
            for u in range(0, NumUsers):
                list_rating = self.pre_MovieLens[u,k_neighbors[i]].astype('float64') #rating list
            
                b_ui = mean + b_u[u] + b_i[i] #user u, item i
                b_uj = mean + b_u[u] + b_i[k_neighbors[i]]
            
                #예측값 계산
                denominator = np.sum(list_sim) #분모
                numerator = np.sum(list_sim * (list_rating - b_uj)) #분자
                predicted_rating[u,i] = b_ui + numerator / denominator
            
        return predicted_rating


    def basic_mean_user(self,sim, k):
        self.sim=sim
        self.k=k
        
        predicted_rating = np.array([[0.0 for col in range(500)] for row in range(500)])
    
        mean=np.nanmean(np.where(self.pre_MovieLens!=0,self.pre_MovieLens,np.nan),axis=1)
        
        Sim=sim
        
        k_neighbors=np.argsort(-Sim)
        k_neighbors=np.delete(k_neighbors,np.s_[k:],1) #k개만 남기고 삭제
        
        NumUsers=np.size(self.pre_MovieLens,axis=0) #user size
        NumItems=np.size(self.pre_MovieLens,axis=1) #item size
        
        for u in range(0,NumUsers):
            list_sim=Sim[u,k_neighbors[u]]
            for i in range(0, NumItems):
                list_rating=self.pre_MovieLens[k_neighbors[u],i].astype('float64')
                list_mean=mean[k_neighbors[u]]
            
                denominator = np.sum(list_sim) #분모
                numerator = np.sum(list_sim * (list_rating - list_mean)) #분자
                predicted_rating[u,i] = mean[u] + numerator / denominator
            
        
        return predicted_rating


    def basic_mean_item(self,sim, k):
        self.sim=sim
        self.k=k
        
        predicted_rating = np.array([[0.0 for col in range(500)] for row in range(500)])
        
        mean=np.nanmean(np.where(self.pre_MovieLens!=0,self.pre_MovieLens,np.nan),axis=0) #axis=0 item
        
        self.pre_MovieLens = (self.pre_MovieLens).T
        
        Sim=sim
        
        self.pre_MovieLens = (self.pre_MovieLens).T
        
        k_neighbors=np.argsort(-Sim)
        k_neighbors=np.delete(k_neighbors,np.s_[k:],1) #k개만 남기고 삭제
        
        NumUsers=np.size(self.pre_MovieLens,axis=0) #user size
        NumItems=np.size(self.pre_MovieLens,axis=1) #item size
        
        for i in range(0,NumItems):
            list_sim=Sim[i,k_neighbors[i]]
            for u in range(0, NumUsers):
                list_rating=self.pre_MovieLens[u,k_neighbors[i]].astype('float64')
                list_mean=mean[k_neighbors[i]]
            
                denominator = np.sum(list_sim) #분모
                numerator = np.sum(list_sim * (list_rating - list_mean)) #분자
                predicted_rating[u,i] = mean[i] + numerator / denominator
            
        
        return predicted_rating

    
    def basic_zscore_user(self,sim, k):
        self.sim=sim
        self.k=k
        
        #예측값 넣을 빈 행렬 생성
        predicted_rating = np.array([[0.0 for col in range(500)] for row in range(500)])
        
        #평균 & 표준편차 - 0값은 NaN값으로 변환
        mean = np.nanmean(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan), axis = 1)
        std = np.nanstd(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan), axis = 1)
        
        #Similarty 불러오기
        Sim=sim
        
        #상위 K명 구하기
        k_neighbors = np.argsort(-Sim) 
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1) #k개만 남기고 삭제

        NumUsers = np.size(self.pre_MovieLens, axis = 0) #user size
        NumItems = np.size(self.pre_MovieLens, axis = 1) #item size
        
        for u in range(0, NumUsers):
            list_sim = Sim[u, k_neighbors[u]] #similarity list
            for i in range (0, NumItems):
                list_rating = self.pre_MovieLens[k_neighbors[u],i].astype('float64') #rating list
                list_mean = mean[k_neighbors[u]] #mean list
                list_std = std[k_neighbors[u]] #std list
            
                #예측값 계산
                denominator = np.sum(list_sim) #분모
                numerator = np.sum(list_sim * ((list_rating - list_mean)/ list_std)) #분자
                predicted_rating[u,i] = mean[u] + std[u] * numerator / denominator
            
        return predicted_rating


    def basic_zscore_item(self,sim, k):
        self.sim=sim
        self.k=k
        
        #예측값 넣을 빈 행렬 생성
        predicted_rating = np.array([[0.0 for col in range(500)] for row in range(500)])
        
        #평균 & 표준편차 - 0값은 NaN값으로 변환
        mean = np.nanmean(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan), axis = 0)
        std = np.nanstd(np.where(self.pre_MovieLens != 0, self.pre_MovieLens, np.nan), axis = 0)
        
        self.pre_MovieLens=(self.pre_MovieLens).T
        
        #Similarty 불러오기
        Sim=sim
        
        self.pre_MovieLens=(self.pre_MovieLens).T
        
        #상위 K명 구하기
        k_neighbors = np.argsort(-Sim) 
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1) #k개만 남기고 삭제

        NumUsers = np.size(self.pre_MovieLens, axis = 0) #user size
        NumItems = np.size(self.pre_MovieLens, axis = 1) #item size
        
        for i in range(0, NumItems):
            list_sim = Sim[i, k_neighbors[i]] #similarity list
            for u in range (0, NumUsers):
                list_rating = self.pre_MovieLens[u,k_neighbors[i]].astype('float64') #rating list
                list_mean = mean[k_neighbors[i]] #mean list
                list_std = std[k_neighbors[i]] #std list
            
                #예측값 계산
                denominator = np.sum(list_sim) #분모
                numerator = np.sum(list_sim * ((list_rating - list_mean)/ list_std)) #분자
                predicted_rating[u,i] = mean[u] + std[u] * numerator / denominator
            
        return predicted_rating
