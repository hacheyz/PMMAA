|     索引      |           名称           |                             概述                             |                           核心代码                           |
| :-----------: | :----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     3.1.1     |         差分方程         |                        求递推数列通项                        |                    `sp.rsolve()` 直接求解                    |
|     3.1.2     | 莱斯利 (Leslie) 种群模型 |        已知初始状态、生育率、存活率<br/>预测种群总量         | $\small \bm A = \bm {PD} \bm P^{-1}$<br/>`P, D = sp.Matrix(ins=A).diagonalize()`<br/>特征值分解\|相似对角化<br/>matplotlib 刻度与刻度标签控制 |
|     3.1.3     |      PageRank 算法       |               图论、马尔可夫链、互达的概率排序               |             有向图构造<br/>`ax.bar()` 柱状图绘制             |
| $\rightarrow$ |       随机冲浪模型       |                     非互达，引入阻尼因子                     |                              -                               |
|     3.2.2     |       推荐系统评分       |                 基于皮尔逊相关系数的评分预测                 | `np.corrcoef()` 按行返回相关系数矩阵<br/>`ax.imshow()` 热力图绘制 |
| $\rightarrow$ |  基于奇异值分解压缩数据  |                 稀疏矩阵降维<br/>余弦相似度                  | $\small \bm A = \bm U\begin{bmatrix}\bm \Sigma 	&\bm 0\\ \bm 0		&\bm 0\end{bmatrix}\bm V^{\rm T}$<br/>`U, S, VT = np.linalg.svd(A)`<br/>其中 `np.diag(S)`=$\small \begin{bmatrix}\bm \Sigma 	&\bm 0\\ \bm 0		&\bm 0\end{bmatrix}$<br/>列压缩数据降维范式 |
|     3.2.2     |   利用SVD进行图像压缩    |                    只保留那些较大的奇异值                    | 数字图像处理库 PIL.Image<br/>妙用`ax.imshow(cmap='gray')`绘制灰度图<br/>对矩阵整体进行数据压缩范式 |
|     4.1.2     |    线性规划模型 (LP)     | ● 企业安排生产问题<br/>● 项目投资问题<br/>● 仓库租借问题<br/>● 最小费用运输问题 | `import cvxpy as cp`<br/>`x = cp.Variable()`<br/>`obj = cp.Maximize[Minimize]()`<br/>`cons = [...]`<br/>`prob = cp.Problem(obj, cons)`<br>`prob.solve(solver='...')` |
|     4.2.1     |     0-1整数规划模型      |      ● 背包问题<br />● 指派问题<br />● 旅行商问题 (TSP)      |           *指派问题代码见4hw5*<br />*TSP代码见4.4*           |
|     4.2.2     |       整数规划模型       | ● 工时安排问题<br/>● 装修分配任务问题 (非标准指派问题)<br/>● 网点覆盖问题 (双决策变量) | `sklearn.metrics.euclidean_distances`<br/>两个矩阵的成对平方欧氏距离 |
|      4.3      |      多目标规划模型      | ● 组合投资问题：<br/>将可供投资的资金分成$n+1$份，<br/>分别购买$n+1$种资产，<br/>同时兼顾投资的净收益和风险 | 多目标模型的目标函数线性化<br />引入变量$x_{n+1}={\rm max} \left\{q_ix_i\right\}$ |
|      4.4      |        旅行商模型        |         ● 比赛项目排序问题<br />引入虚拟项目构成闭环         |           `NaN`的处理：`data[np.isnan(data)] = 0`            |
|      4hw      |            -             | ● 限制运量的最小费用运输问题<br />● 面试顺序问题<br />● 带约束的背包问题<br />● 钢管下料问题 |                              -                               |
|               |                          |                                                              |                                                              |
