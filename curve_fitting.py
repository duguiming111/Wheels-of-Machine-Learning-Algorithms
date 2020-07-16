# Author: dgm
# Description: 多项式拟合
# Date: 2020-07-16
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def real_func(x):
    """目标函数"""
    return np.sin(2*np.pi*x)


def fit_func(p, x):
    """多项式"""
    f = np.poly1d(p)
    return f(x)


def residuals_func(p, x, y):
    """计算残差"""
    ret = fit_func(p, x) - y
    return ret


def residuals_func_regularization(p, x, y):
    """加正则化的残差"""
    # 正则化参数
    regularization = 0.001
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))  # L2范数作为正则化项
    return ret


def fitting(x, y, x_points, M=0):
    """多项式拟合"""
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    # p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    # 带正则化最小二乘法
    p_lsq = leastsq(residuals_func_regularization, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq


if __name__ == "__main__":
    # 一 构造数据
    x = np.linspace(0, 1, 10)
    x_points = np.linspace(0, 1, 1000)
    # 加上正态分布噪音的目标函数的值
    y_ = real_func(x)
    y = [np.random.normal(0, 0.1) + y1 for y1 in y_]

    # 二 拟合
    p_lsq = fitting(x, y, x_points, M=9)
