import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman + SimSun + WFM Sans SC']
plt.rcParams['mathtext.fontset']='stix'
# Times New Roman + SimSun + WFM Sans SC
# simsum宋体, times new roman -*, simhei黑体, kaiti楷体, 
# dengxian等线, fangsong仿宋, Microsoft Yahei微软雅黑
plt.rcParams['axes.unicode_minus']=False   
plt.rcParams['figure.dpi'] = 200
# plt.rcParams['figure.figsize'] = [4, 3]
# plt.rcParams['font.size'] = 12
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

pos = np.array([
    [150, 140, 243],
    [85, 85, 236],
    [150, 155, 220.5],
    [145, 50, 159],
    [130, 150, 230],
    [0, 0, 52],
])
pos[:, 2] = pos[:, 2]/180*np.pi # 化为弧度制
pos

from sklearn.metrics import euclidean_distances
loc = pos[:, :2]
a, b = np.zeros((6, 6)), np.zeros((6,6))
r = euclidean_distances(loc, loc)
min_distance = 8
plane_num = 6

# a = np.arcsin(min_distance/r)
for i in range(plane_num):
    for j in range(plane_num):
        if i != j:
            a[i, j] = np.arcsin(min_distance/r[i, j])

# 化回到角度制
a = a*180/np.pi
np.round(a, 4)

# b = np.angle()
c_alg = loc[:, 0] + loc[:, 1]*1j # 复数的代数形式
c_exp = np.exp(1j*pos[:, 2])        # 复数的指数形式
for m in range(plane_num):
    for n in range(plane_num):
        if m != n:
            b[m, n] = np.angle((c_exp[n] - c_exp[m]) / (c_alg[m] - c_alg[n]))
            # 复数一旦形成，均被视为代数形式

# 化回到角度制
b = b*180/np.pi
np.round(b, 4)

obj = lambda x: np.sum(np.abs(x))
x0 = [0,2,100,6,0,0]  # 决策变初值的选定
cons=[]
for i in range(plane_num):
        for j in range(i+1, plane_num):
            cons.append({'type': 'ineq', 'fun': lambda x: np.abs(b[i,j] + (x[i]+x[j])/2) - a[i,j]})
bd = [(-30, 30) for _ in range(plane_num)]
ret = minimize(obj, x0, constraints=cons, bounds=bd)
print(ret)
print('-'*80)
print('目标函数的最优值：', round(ret.fun, 4))
print('最优解为：', np.round(ret.x, 4))