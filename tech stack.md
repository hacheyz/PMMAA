|     索引      |           名称           |                             概述                             |                           核心代码                           |
| :-----------: | :----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     3.1.1     |         差分方程         |                        求递推数列通项                        |                    `sp.rsolve()` 直接求解                    |
|     3.1.2     | 莱斯利 (Leslie) 种群模型 |        已知初始状态、生育率、存活率<br/>预测种群总量         | $\small \bm A = \bm {PD} \bm P^{-1}$<br/>`P, D = sp.Matrix(ins=A).diagonalize()`<br/>特征值分解\|相似对角化<br/>matplotlib 刻度与刻度标签控制 |
|     3.1.3     |      PageRank 算法       |               图论、马尔可夫链、互达的概率排序               |             有向图构造<br/>`ax.bar()` 柱状图绘制             |
| $\rightarrow$ |       随机冲浪模型       |                     非互达，引入阻尼因子                     |                              -                               |
|     3.2.2     |       推荐系统评分       |                 基于皮尔逊相关系数的评分预测                 | `np.corrcoef()` 按行返回相关系数矩阵<br/>`ax.imshow()` 热力图绘制 |
| $\rightarrow$ |  基于奇异值分解压缩数据  |                 稀疏矩阵降维<br/>余弦相似度                  | $\small \bm A = \bm U\begin{bmatrix}\bm \Sigma 	&\bm 0\\\bm 0		&\bm 0\end{bmatrix}\bm V^{\rm T}$<br/>`U, S, VT = np.linalg.svd(A)`<br/>其中 `np.diag(S)`=$\small \begin{bmatrix}\bm \Sigma 	&\bm 0\\\bm 0		&\bm 0\end{bmatrix}$<br/>列压缩数据降维范式 |
|     3.2.2     |   利用SVD进行图像压缩    |                    只保留那些较大的奇异值                    | 数字图像处理库 PIL.Image<br/>妙用`ax.imshow(pcmap='gray')`绘制灰度图<br/>对矩阵整体进行数据压缩范式 |
|     4.1.2     |    线性规划模型 (LP)     | 企业安排生产问题<br/>项目投资问题<br/>仓库租借问题<br/>最小费用运输问题 | `import cvxpy as cp`<br/>`x = cp.Variable()`<br/>`obj = cp.Maximize[Minimize]()`<br/>`cons = [...]`<br/>`prob = cp.Problem(obj, cons)`<br>`prob.solve(solver='...')` |
|     4.2.1     |     0-1整数规划模型      |                背包问题、指派问题、旅行商问题                |                              -                               |
|     4.2.2     |       整数规划模型       | 工时安排问题<br/>装修分配任务问题 (非标准指派问题)<br/>网点覆盖问题 (双决策变量) |                                                              |
