#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

class Preprocessing_RSData:
    def __init__(self,matrix):
        self.matrix=matrix
    
    def Preprocessing(self):
        Num_Mat_MovieLens=np.array(self.matrix)
        
        pre_MovieLens=pd.DataFrame(Num_Mat_MovieLens) #전치 했었는데 없앴음 히히
        
        ##### item 기준 전처리 #####
        # rating된 값이 많은 순서대로 index값 추출
        # 상위 1000개를 제외하고 item_index에 저장
        item_index=pre_MovieLens[pre_MovieLens!=0].count().sort_values(ascending=False).index[500:]
        # artist_index를 제외한 상위 1000개의 artist_id만 남겨놓음
        pre_MovieLens=pre_MovieLens.drop(item_index, axis=1)


        ##### user 기준 전처리 #####
        # rating된 값이 많은 순서대로 index값 추출
        # 상위 1000개를 제외하고 user_index에 저장
        user_index=pre_MovieLens[pre_MovieLens!=0].count(axis=1).sort_values(ascending=False).index[500:]
        # user_index를 제외한 상위 1000개의 user_id만 남겨놓음
        pre_MovieLens=pre_MovieLens.drop(user_index, axis=0)


        # index 순서대로 sorting
        pre_MovieLens=pre_MovieLens.sort_index()
        pre_MovieLens=pre_MovieLens.sort_index(axis=1)
   
        return pre_MovieLens
