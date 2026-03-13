import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass
from botorch.models.transforms.input import InputTransform
from typing import Optional, Union

class RectangleSphericalVariablesChange:
    def __init__(self, do_x_sqrt:bool=True, return_r:bool=True):
        '''
        垂直坐標系與球坐標系之間的變數變換
        do_x_sqrt:bool = True
        => 是否需要幫輸入的 X 開根號後再做變數變換成極座標？ U = sqrt(X)

        return_r:bool = True 
        => to_theta 的 function 中：回傳的極座標資料是否需要包含 r，如果有包含的話第1個 column 就是 r
        => to_x 的 function 中：輸入的極座標資料是否有包含 r，如果沒有的話就在第一行插入1
        '''
        self.do_x_sqrt = do_x_sqrt
        self.return_r = return_r

    def to_theta(self, data:Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> torch.Tensor:
        '''
        直角坐標系轉成球坐標系
        
        :param data: 直角坐標系資料
        :type data: Union[np.ndarray, torch.Tensor, pd.DataFrame]
        :return: 球坐標系資料
        :rtype: Tensor
        '''
        device = torch.get_default_device()
        dtype = torch.get_default_dtype()
        
        if isinstance(data, np.ndarray):
            X = torch.tensor(data, device=device, dtype=dtype)
        elif isinstance(data, torch.Tensor):
            X = data
        elif isinstance(data, pd.DataFrame):
            X = torch.tensor(np.array(data), device=device, dtype=dtype)

        # 如果希望做極座標轉換的 X 是原本的 X 開根號
        X = torch.sqrt(X) if self.do_x_sqrt else X

        # 所有維度平方相加開根號, 並且做成一個 Nx1 的矩陣
        r = torch.sqrt(torch.sum(torch.square(X), dim=1)).unsqueeze(1)

        # 計算 theta 的數量
        num_theta = X.shape[1] - 1

        # 計算每個 theta
        thetas = []
        for i in range(num_theta):
            theta_idx = i + 1
            theta = torch.atan2( torch.sqrt( torch.sum( torch.square(X[:,theta_idx:]), dim=1 ) ), X[:,i] ).unsqueeze(1)
            thetas.append(theta)

        thetas = torch.concat(thetas, dim=1)

        polor_data = torch.concat([r, thetas], dim=1) if self.return_r else thetas

        return polor_data
    
    def to_x(self, polor_data:torch.Tensor) -> torch.Tensor:
        '''
        球坐標系資料轉成直角坐標系資料
        
        :param polor_data: 球坐標系資料
        :type polor_data: torch.Tensor
        :return: 直角坐標系資料
        :rtype: Tensor
        '''
        device = polor_data.device
        dtype = torch.get_default_dtype()

        
        if self.return_r: # 如果輸入的資料包含 r
            r = polor_data[:,:1]
            thetas = polor_data[:,1:]
        else: # 如果輸入的資料不包含 r
            r = torch.ones([polor_data.shape[0], 1], device=device)
            thetas = polor_data

        X = []
        sin_prod = torch.ones((thetas.shape[0], 1), device=device, dtype=dtype)
        for theta_idx in range(thetas.shape[1]):
            theta_i = thetas[:, theta_idx:theta_idx+1]
            cos_theta = torch.cos(theta_i).to(device)
            sin_theta = torch.sin(theta_i).to(device)
            x = r * sin_prod * cos_theta
            X.append(x)
            sin_prod = sin_prod * sin_theta # 為下一個 theta 轉 X 做準備
        
        # 計算最後一個維度
        x_n = r*sin_prod
        X.append(x_n)

        X = torch.concat(X, dim=1)

        # 如果當初的 X 有開根號的話就必須還原
        X = X**2 if self.do_x_sqrt else X

        return X
    
    @staticmethod
    def describe(data:torch.Tensor) -> pd.DataFrame:
        '''
        列出data 中每個 column 的平均數、標準差、最小值、第一二三四分位數、最大值、 column 中數值為0的比例
        :param data: 需要統計的資料
        :type data: torch.Tensor
        '''
        describe = pd.DataFrame(data.cpu().numpy()).describe()
        zero_ratio = (data == 0).float().mean(dim=0).cpu().numpy().round(3)

        describe.loc[len(describe)] = zero_ratio
        new_index = describe.index.tolist()
        new_index[-1] = "zero_ratio"
        describe.index = new_index

        return describe.round(3)
    
class CartesianToSpherical(nn.Module):
    def __init__(self, do_x_sqrt:bool=True, return_r:bool=True, eps=1e-8):
        super(CartesianToSpherical, self).__init__()
        self.do_x_sqrt = do_x_sqrt
        self.return_r = return_r
        self.eps = eps  # 用於防止除以零

    def forward(self, X:torch.Tensor):
        '''
        直角坐標系轉成球坐標系
        
        :param data: 直角坐標系資料
        :type X: torch.Tensor
        :return: 球坐標系資料
        :rtype: Tensor
        '''
        device = X.device
        # device = torch.get_default_device()
        # dtype = torch.get_default_dtype()

        # 如果希望做極座標轉換的 X 是原本的 X 開根號
        X = torch.sqrt(X) if self.do_x_sqrt else X

        # 所有維度平方相加開根號, 並且做成一個 Nx1 的矩陣
        r = torch.sqrt(torch.sum(torch.square(X), dim=1)).unsqueeze(1)

        # 計算 theta 的數量
        num_theta = X.shape[1] - 1

        # 計算每個 theta
        thetas = []
        for i in range(num_theta):
            theta_idx = i + 1
            theta = torch.atan2( torch.sqrt( torch.sum( torch.square(X[:,theta_idx:]), dim=1 ) ), X[:,i] ).unsqueeze(1)
            thetas.append(theta)

        thetas = torch.concat(thetas, dim=1)

        polor_data = torch.concat([r, thetas], dim=1) if self.return_r else thetas

        return polor_data
    
class SphericalToCartesian(nn.Module):
    def __init__(self, do_x_sqrt:bool=True, return_r:bool=True, eps=1e-8):
        super().__init__()
        self.do_x_sqrt = do_x_sqrt
        self.return_r = return_r
        self.eps = eps  # 用於防止除以零

    def forward(self, X:torch.Tensor):

        device = X.device
        if self.return_r: # 如果輸入的資料包含 r
            r = X[:,:1]
            thetas = X[:,1:]
        else: # 如果輸入的資料不包含 r
            r = torch.ones([X.shape[0], 1], device=device)
            thetas = X
            
        Xc = []
        sin_prod = torch.ones((thetas.shape[0], 1), device=device)
        for theta_idx in range(thetas.shape[1]):
            theta_i = thetas[:, theta_idx:theta_idx+1]
            cos_theta = torch.cos(theta_i).to(device)
            sin_theta = torch.sin(theta_i).to(device)
            xc = r * sin_prod * cos_theta
            Xc.append(xc)
            sin_prod = sin_prod * sin_theta # 為下一個 theta 轉 X 做準備
        
        # 計算最後一個維度
        xc_n = r*sin_prod
        Xc.append(xc_n)

        Xc = torch.concat(Xc, dim=1)

        # 如果當初的 X 有開根號的話就必須還原
        Xc = Xc**2 if self.do_x_sqrt else Xc

        return Xc

class SphericalModelWrapper(nn.Module):
    def __init__(self, base_model, transform):
        super().__init__()
        self.base_model = base_model
        self.input_transform = transform
        
    def posterior(self, X, **kwargs):
        if X.ndim == 3:
            batch, q, d = X.shape
            # 將 batch 和 q 合併，轉換完再拆回來，確保 q 不會跑掉
            X_flat = X.view(-1, d) 
            # 先轉座標，再進模型
            X_transformed = self.input_transform(X_flat)
            # 取得轉換後的特徵維度 d_new
            d_new = X_transformed.shape[-1]
            X_final = X_transformed.view(batch, q, d_new)
        else:
            # 先轉座標，再進模型
            X_final = self.input_transform(X)

        return self.base_model.posterior(X_final, **kwargs)
    
    def forward(self, X, **kwargs):
        if X.ndim == 3:
            batch, q, d = X.shape
            # 將 batch 和 q 合併，轉換完再拆回來，確保 q 不會跑掉
            X_flat = X.view(-1, d) 
            # 先轉座標，再進模型
            X_transformed = self.input_transform(X_flat)
            # 取得轉換後的特徵維度 d_new
            d_new = X_transformed.shape[-1]
            X_final = X_transformed.view(batch, q, d_new)
        else:
            # 先轉座標，再進模型
            X_final = self.input_transform(X)

        # return self.base_model.posterior(X_final, **kwargs)
        return self.base_model.forward(X_final, **kwargs)