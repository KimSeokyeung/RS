#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

class All_Similarity:
    def __init__(self, pre_MovieLens):
        self.pre_MovieLens=pre_MovieLens

    def COS(self):
        self.pre_MovieLens=np.array(self.pre_MovieLens)
        NumUsers = np.size(self.pre_MovieLens, axis=0) # row size
        Sim = np.full((NumUsers, NumUsers), 0.0) # Sim matrix 초기화
    
        for u in range(0, NumUsers): # row : u번째 사용자
            for v in range(u, NumUsers): # column : v번째 아티스트
            
                innerDot = np.dot(self.pre_MovieLens[u, ], self.pre_MovieLens[v, ]) #내적
                NormU = np.linalg.norm(self.pre_MovieLens[u, ]) #user의 크기
                NormV = np.linalg.norm(self.pre_MovieLens[v, ]) #artist의 크기
            
                if(NormU == 0 or NormV ==0): #분모가 0이 되는 경우, 0으로 값 삽입
                    Sim[u,v] = 0
                else: 
                    Sim[u,v] = innerDot/(NormU * NormV) #COS 계산
                
                Sim[v,u] = Sim[u,v]
            
        return Sim

    def COS_except(self):
        self.pre_MovieLens=np.array(self.pre_MovieLens)
        NumUsers = np.size(self.pre_MovieLens, axis=0) # row size
        Sim = np.full((NumUsers, NumUsers), 0.0) # Sim matrix 초기화
    
        for u in range(0, NumUsers): # row : u번째 사용자
            for v in range(u, NumUsers): # column : v번째 아티스트
            
                arridx_u = np.where(self.pre_MovieLens[u, ] == 0)
                arridx_v = np.where(self.pre_MovieLens[v, ] == 0) # NaN값 찾기
                arridx = np.concatenate((arridx_u, arridx_v), axis = None)
            
                U = np.delete(self.pre_MovieLens[u, ], arridx)
                V = np.delete(self.pre_MovieLens[v, ], arridx) # NaN값 제외
            
                innerDot = np.dot(U, V) #내적
                NormU = np.linalg.norm(U) #user의 크기
                NormV = np.linalg.norm(V) #artist의 크기
            
                if(NormU == 0 or NormV == 0): #분모가 0이 되는 경우, 0으로 값 삽입
                    Sim[u,v] = 0
                else: 
                    Sim[u,v] = innerDot/(NormU * NormV) #COS 계산
                
                Sim[v,u] = Sim[u,v]
            
        return Sim

    def PCC(self):
        self.pre_MovieLens=np.array(self.pre_MovieLens)
        NumUsers = np.size(self.pre_MovieLens, axis=0) # row size
        Sim = np.full((NumUsers, NumUsers), 0.0) # Sim matrix 초기화
    
        mean = np.nanmean(np.where(self.pre_MovieLens!=0, self.pre_MovieLens, np.nan), axis=1) #두 사용자 모두 NaN이 아닌 것의 평균
        mean[np.isnan(mean)] = 0
    
        for u in range(0, NumUsers): # row : u번째 사용자
            for v in range(u, NumUsers): # column : v번째 아티스트
            
                arridx_u = np.where(self.pre_MovieLens[u, ] == 0)
                arridx_v = np.where(self.pre_MovieLens[v, ] == 0) # NaN값 찾기
                arridx = np.concatenate((arridx_u, arridx_v), axis = None) #join
               
                U = np.delete(self.pre_MovieLens[u, ], arridx)
                V = np.delete(self.pre_MovieLens[v, ], arridx) # NaN값 제외
            
                U = U - mean[u]
                V = V - mean[v]
            
            # COS과 동일
                innerDot = np.dot(U, V) #내적
                NormU = np.linalg.norm(U) # user의 크기
                NormV = np.linalg.norm(V) # artist의 크기
            
                if(NormU * NormV == 0): #분모가 0이 되는 경우, 0으로 값 삽입
                    Sim[u,v] = 0
                else: 
                    Sim[u,v] = innerDot/(NormU * NormV) #COS 계산
                
                Sim[v,u] = Sim[u,v]
            
        return Sim

    def CPCC(self):
        self.pre_MovieLens=np.array(self.pre_MovieLens)
        NumUsers = np.size(self.pre_MovieLens, axis=0) # row size
        Sim = np.full((NumUsers, NumUsers), 0.0) # Sim matrix 초기화
    
        median = np.nanmedian(np.where(self.pre_MovieLens!=0, self.pre_MovieLens, np.nan), axis=1) #두 사용자 모두 NaN이 아닌 것의 평균
 
        for u in range(0, NumUsers): # row : u번째 사용자
            for v in range(u, NumUsers): # column : v번째 아티스트
            
                arridx_u = np.where(self.pre_MovieLens[u, ] == 0)
                arridx_v = np.where(self.pre_MovieLens[v, ] == 0) # NaN값 찾기
                arridx = np.concatenate((arridx_u, arridx_v), axis = None) #join
               
                U = np.delete(self.pre_MovieLens[u, ], arridx)
                V = np.delete(self.pre_MovieLens[v, ], arridx) # NaN값 제외
            
                U = U - median[u]
                V = V - median[v]
            

            # COS과 동일
                innerDot = np.dot(U, V) #내적
                NormU = np.linalg.norm(U) #user의 크기
                NormV = np.linalg.norm(V) #artist의 크기
            
                Sim[u,v] = innerDot/(NormU * NormV) #COS 계산
                Sim[v,u] = Sim[u,v]
                Sim[np.isnan(Sim)] = -1 #NaN값 제거
            
        return Sim

    def JAC(self):
        self.pre_MovieLens=np.array(self.pre_MovieLens)
        NumUsers = np.size(self.pre_MovieLens, axis=0) # row size
        Sim = np.full((NumUsers, NumUsers), 0.0) # Sim matrix 초기화
        
        for u in range(0, NumUsers): # row : u번째 사용자
            for v in range(u, NumUsers): # column : v번째 아티스트
                
                U = np.array(self.pre_MovieLens[u, ] > 0, dtype = np.int)
                V = np.array(self.pre_MovieLens[v, ] > 0, dtype = np.int) # rating binary
            
                SumUV = U + V 
            #합 = 2 : 둘 다 rating / 합 = 1 : 둘 중 하나만 rating / 합 = 0 : 둘 다 rating X
            
                Inter = np.sum(np.array(SumUV > 1, dtype = np.int)) #교집합
                Union = np.sum(np.array(SumUV > 0, dtype = np.int)) #합집합
            
                if(Union == 0): #분모가 0이 되는 경우, 0으로 값 삽입
                    tmp=0
                else:    
                    tmp = Inter / Union # 교집합/합집합
                Sim[u,v] = tmp
                Sim[v,u] = Sim[u,v]
            
        return Sim

    def EUC(self):
        self.pre_MovieLens=np.array(self.pre_MovieLens)
        NumUsers = np.size(self.pre_MovieLens, axis=0) # = row size
        Sim = np.full((NumUsers, NumUsers), 0.0) #Sim matrix 초기화
    
        for u in range(0, NumUsers): # u번째 사용자
            for v in range(u, NumUsers): # v번째 아티스트
                
                tmp = np.sum(np.square(self.pre_MovieLens[u,]-self.pre_MovieLens[v,])) #EUC값 계산
                
                Sim[u,v] = np.sqrt(tmp)
                Sim[v,u] = Sim[u,v]
            
        return Sim

    def MSD(self):
        self.pre_MovieLens=np.array(self.pre_MovieLens)
        NumUsers = np.size(self.pre_MovieLens, axis=0) # row size
        Sim = np.full((NumUsers, NumUsers), 0.0) #sim matrix 초기화
    
        for u in range(0, NumUsers): # u번째 사용자
            for v in range(u, NumUsers): #v번째 아티스트
            
                U = np.where(self.pre_MovieLens[u, ] == 0, np.nan, self.pre_MovieLens[u, ])
                V = np.where(self.pre_MovieLens[v, ] == 0, np.nan, self.pre_MovieLens[v, ]) # 0을 NaN으로 
            
                SquaredSum = np.square(U - V) #MSD값 계산
                SquaredSum = SquaredSum[~np.isnan(SquaredSum)] # Nan값 제거
            
                AllItems = np.size(SquaredSum, axis = 0) #NaN값을 제외한 크기
            
                if(AllItems == 0):
                    tmp = 0
                else:
                    tmp = np.sum(SquaredSum)/AllItems #Nan값을 제외한 합
                
                Sim[u,v] = tmp
                Sim[v,u] = Sim[u,v]
            
        return Sim

    def JMSD (self, max):
        self.pre_MovieLens=np.array(self.pre_MovieLens)
        return JAC(self.pre_MovieLens) * (1-(MSD(self.pre_MovieLens/max)))
    
