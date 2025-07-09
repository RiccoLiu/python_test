from __future__ import print_function
import torch

from loguru import logger

'''
torch.Tensor:
    .requires_grad=True:
        属性为True时会开始跟踪针对 tensor 的所有操作, 该张量的梯度将累积到 .grad 属性中。

    .grad_fn:        
        保存着创建了张量的 Function 的引用，（如果用户自己创建张量，则g rad_fn 是 None ）

    .detach():
        它将其与计算历史记录分离，停止计算梯度

    .backward():
        自动针对tensord的梯度, 如果tensor是标量，则不需要指定参数；如果有更多元素，则需要指定一个 gradient 参数来指定张量的形状;

    .requires_grad_(True):
        可以改变张量的 requires_grad 属性

    with torch.no_grad()
        接下来的操作不会计算梯度
'''

if __name__ == '__main__':
    
    '''
        grad_fn:
            表示张量的计算历史记录， 
    '''
    
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    
    logger.info(y)
    logger.info(y.grad_fn)
 
    z = y * y * 3
    logger.info(z)

    out = z.mean()
    logger.info(out)
    
    
    '''    
        requires_grad_:
            改变张量的 requires_grad 属性，
    '''
    
    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    logger.info(a.requires_grad)
    
    a.requires_grad_(True)
    logger.info(a.requires_grad)

    b = (a * a).sum()
    logger.info(b.grad_fn)
    
    '''
        反向传播:
    '''
    
    out.backward() # 等价与 out.backward(torch.tensor(1.))。
    
    # 打印梯度 1/ 4 Sigma 3*(x+2)^2
    logger.info('x.grad:\n', x.grad)

    '''
        雅可比向量积：
            没明白咋回事
    '''
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    logger.info(y)

    
    