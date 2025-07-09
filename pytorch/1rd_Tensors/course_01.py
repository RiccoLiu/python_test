
from __future__ import print_function
import torch

from loguru import logger


'''
Tensors:
    初始化：
        x = torch.empty(5,3)
            5行3列矩阵，未初始化

        x = torch.rand(5, 3)
            5行3列的矩阵，随机初始化

        x = torch.zeros(5, 3)
            5行3列的矩阵，全0初始化
        
        x = torch.tensor([[5, 3], [2, 1]]， dtype=torch.long)
            5行3列的矩阵，指定元素和类型初始化
    
    打印维度：
        print(x.size()) # 返回一个元组
        print(x.shape)  # 返回一个元组
    
    加法：
        基本用法：
            res = x+y
            res = torch.add(x, y)
            res = torch.zeros(2, 2); torch.add(x, y, out=res)
            
        in-place:
            任何使张量会发生变化的操作都有一个前缀 '_'， 比如：
            y.add_(x):         
            y.copy_(y):
            y.t_():
    
    改变维度：
        x = x.view(16); logger.info(x.shape)
        x = x.view(2, 8); logger.info(x.shape)
        x = x.view(-1, 4); logger.info(x.shape)
    
    获取张量的值：    
        x = torch.randn(1)
        logger.info(x)
        logger.info(x.item())
        logger.info(x[0])
'''

if __name__ == '__main__':
    '''
        构建一个5行3列的矩阵，不初始化
    '''
    x = torch.empty(5, 3)
    logger.info(x)
    
    '''
        构造一个随机初始化的矩阵：
    '''
    x = torch.rand(5, 3)
    logger.info(x)
    
    '''
        构造一个全零矩阵
    '''
    x = torch.zeros(5, 3, dtype=torch.long)
    logger.info(x)
    
    '''
        指定数据初始化
    '''
    x = torch.tensor([[5, 3],
                      [2, 1]])
    
    logger.info(x)

    '''
        打印tensor 维度
    '''
    logger.info(x.size())
    logger.info(x.shape)

    rows, cols = x.shape
    logger.info(f'rows:{rows}, cols:{cols}')

    '''
        tensor 加法
    '''
    y = torch.rand(2, 2)
    
    logger.info(x + y) 
    logger.info(torch.add(x, y))

    result = torch.zeros(2, 2); torch.add(x, y, out=result)
    logger.info(result)

    '''
        in-place
    '''
    res = torch.zeros(2, 2)
    res.add_(x)
    res.add_(y)
    logger.info(res)

    '''
        改变维度
    '''
    x = torch.randn(4, 4)

    x = x.view(16); logger.info(x.shape)
    x = x.view(2, 8); logger.info(x.shape)
    x = x.view(-1, 4); logger.info(x.shape)

    '''
        item()
            获取张量的值
    '''
    x = torch.randn(1)
    logger.info(x)
    logger.info(x.item())
    logger.info(x[0])



    x = torch.rand(1080, 1920, 1, 1)
    print(f'x.shape:{x.shape}')

    x = x.squeeze()
    
    print(f'x.shape:{x.shape}')

