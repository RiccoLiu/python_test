
import numpy as np
from sklearn.linear_model import LinearRegression

def get_angle(xs, y_samples):
    '''
        线形回归拟合直线计算与y轴的角度，取值范围[-PI / 2, PI / 2]
    '''
    lr = LinearRegression()
    xs, ys = xs[xs >= 0], y_samples[xs >= 0] # 计算 
    if len(xs) > 1:
        lr.fit(ys[:, None], xs) # 计算 xs = k * ys + b
        k = lr.coef_[0]
        theta = np.arctan(k) # y轴角度
    else:
        theta = 0
    return theta
