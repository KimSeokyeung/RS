#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

#파일을 읽어 행렬에 저장하기 위한 Class
class CallRSData:
    def __init__(self, filepath):
        self.filepath=filepath
        
    def CallMovieLens(self):
        matrix=[[0 for col in range(1682)] for row in range(943)]
        
        f=open(self.filepath, "r")
        
        while True:
            line=f.readline()
            
            if not line:
                break
                
            line_split=line.split('\t')
            
            matrix[int(line_split[0])-1][int(line_split[1])-1]=int(line_split[2])
            
        f.close
        
        return matrix