|       索引       |           名称           |                             概述                             |                           核心代码                           |
| :--------------: | :----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      3.1.1       |         差分方程         |                        求递推数列通项                        |                    `sp.rsolve()` 直接求解                    |
|      3.1.2       | 莱斯利 (Leslie) 种群模型 |        已知初始状态、生育率、存活率<br/>预测种群总量         | $\small \bm A = \bm {PD} \bm P^{-1}$<br/>`P, D = sp.Matrix(ins=A).diagonalize()`<br/>特征值分解\|相似对角化<br/>matplotlib 刻度与刻度标签控制 |
|      3.1.3       |      PageRank 算法       |               图论、马尔可夫链、互达的概率排序               |             有向图构造<br/>`ax.bar()` 柱状图绘制             |
|  $\rightarrow$   |       随机冲浪模型       |                     非互达，引入阻尼因子                     |                              -                               |
|      3.2.2       |       推荐系统评分       |                 基于皮尔逊相关系数的评分预测                 | `np.corrcoef()` 按行返回相关系数矩阵<br/>`ax.imshow()` 热力图绘制 |
|  $\rightarrow$   |  基于奇异值分解压缩数据  |                 稀疏矩阵降维<br/>余弦相似度                  | $\small \bm A = \bm U\begin{bmatrix}\bm \Sigma 	&\bm 0\\ \bm 0		&\bm 0\end{bmatrix}\bm V^{\rm T}$<br/>`U, S, VT = np.linalg.svd(A)`<br/>其中 `np.diag(S)`=$\small \begin{bmatrix}\bm \Sigma 	&\bm 0\\ \bm 0		&\bm 0\end{bmatrix}$<br/>列压缩数据降维范式 |
|      3.2.2       |   利用SVD进行图像压缩    |                    只保留那些较大的奇异值                    | 数字图像处理库 PIL.Image<br/>妙用`ax.imshow(cmap='gray')`绘制灰度图<br/>对矩阵整体进行数据压缩范式 |
|      4.1.2       |    线性规划模型 (LP)     | ● 企业安排生产问题<br/>● 项目投资问题<br/>● 仓库租借问题<br/>● 最小费用运输问题 | `import cvxpy as cp`<br/>`x = cp.Variable()`<br/>`obj = cp.Maximize[Minimize]()`<br/>`cons = [...]`<br/>`prob = cp.Problem(obj, cons)`<br>`prob.solve(solver='...')` |
|      4.2.1       |     0-1整数规划模型      |           背包问题、指派问题<br />旅行商问题 (TSP)           |          *指派问题代码见4hw.5*<br />*TSP代码见4.4*           |
|      4.2.2       |       整数规划模型       | ● 工时安排问题<br/>● 装修分配任务问题 (非标准指派问题)<br/>● 网点覆盖问题 (双决策变量) | `sklearn.metrics.euclidean_distances`<br/>两个矩阵的成对平方欧氏距离 |
|       4.3        |      多目标规划模型      | ▲ 组合投资问题：<br/>将可供投资的资金分成 $n\!+\!1$ 份，<br/>分别购买 $n\!+\!1$ 种资产，<br/>同时兼顾投资的净收益和风险<br /> | 多目标模型的目标函数线性化<br />[优化模型线性化方法总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/361766549)<br />引入变量$x_{n+1}={\rm max} \left\{q_ix_i\right\}$<br />投资风险为总收益的方差 |
|       4.4        |        旅行商模型        |         ▲ 比赛项目排序问题<br />引入虚拟项目构成闭环         |           `NaN`的处理：`data[np.isnan(data)] = 0`            |
|       4hw        |            -             | ● 限制运量的最小费用运输问题<br />● 面试顺序问题<br />● 带约束的背包问题<br />● 钢管下料问题 |                              -                               |
|       5.2        |      非线性规划模型      |                        ▲ 彩电生产问题                        | 灵敏性分析相关计算和表述<br />`sympy`模块`diff`，`subs`等函数的使用<br />三维图坐标刻度控制 |
|       5.3        |       二次规划模型       | 目标函数为决策向量的二次函数<br />约束条件均为线性的<br />▲ 投资组合 (portfolio) 问题 | `cp.quad_form(x, c)` 返回二次型<br />`np.cov()` 按行返回协方差矩阵 |
|       5.4        |       非凸规划模型       |                              -                               | `scipy.optimize.minimize()`局部最优解<br />*先用 `cvxpy` 求解，发现非凸后再用 `minimize`<br />*另外，对于无约束问题，可以采用<br />`scipy.optimize.basinhopping()`<br />获得全局最优解 |
|  $\rightarrow$   |            -             |                       ● 供应与选址问题                       |             `minimize` 多元决策变量的划分与解包              |
|       5.5        |      多目标规划模型      |                       ▲ 生产与污染问题                       |                              -                               |
|       5.6        |            -             |                        ▲ 飞行管理问题                        | 绘制箭头 `arrow()`<br />相对运动、三角学<br />`numpy` 模块中的复数、辐角 `angle()`<br />Python特性：延迟绑定 |
|       5hw6       |            -             |                     ● 组合投资问题<br />                     |                              -                               |
|       5hw7       |            -             |                        ● 生产计划问题                        |                        `cp.cumsum()`                         |
| 6.3.1<br />6.3.2 |        最短路算法        |                  Dijkstra 算法、Floyd 算法                   | `nx.dijkstra_path()`<br />`nx.dijkstra_path_length()`<br />`nx.shortest_path()`<br />`nx.shorted_path_length`<br />`nx.floyd_warshall_numpy()` |
|      6.3.3       |        最短路应用        |                     ● 设备更新问题<br />                     |                     `nx.shortest_path()`                     |
|      6.3.3       |        最短路应用        |                       ● 选址问题<br />                       |                 `np.argmin()`，`np.argmax()`                 |
|      6.3.4       |            -             |                  最短路问题的 0-1 规划模型                   |                              -                               |
|      6.4.1       |        最小生成树        |                        ● 架设电线问题                        |                 `nx.minimum_spanning_tree()`                 |
|      6.4.2       |            -             |                最小生成树问题的 0-1 规划模型                 |                              -                               |
|       6.5        |         着色问题         | 物资储存问题<br />无线交换设备的波长分配问题<br />● 会议安排问题 |                  计算色数的整数线性规划模型                  |
|      6.6.1       |        最大流问题        |       最大流问题的 0-1 规划模型<br />● 多对多招聘问题        |      `nx.maximum_flow()`<br />从字典中导出邻接矩阵范式       |
|      6..6.2      |      最小费用流问题      |              ● 运费网络模型<br />最小费用最大流              |      `nx.max_flow_min_cost()`<br />`nx.cost_of_flow()`       |
|      6.7.4       |    计划网络的优化问题    |                          ● 赶工问题                          |                              -                               |
|      6.7.5       |            -             |               求完成作业的期望和实现事件的概率               |   `scipy.stats.norm` (正态分布)<br />中的 `pdf()`，`cdf()`   |
|       6.8        |            -             |                     ▲ 钢管订购和运输问题                     |                 `nx.floyd_warshall_numpy()`                  |
|       6hw4       |            -             |                   ● 允许售出的设备更新问题                   |                     `nx.shortest_path()`                     |
|       6hw7       |            -             |                      无向图的度量图绘制                      |                  `ax.plot()` 可绘制顶点和边                  |
|      7.1.1       |         一维插值         | 多项式插值<br />拉格朗日插值法<br />分段线性插值、三次样条插值 | `np.vander()` 返回范德蒙行列式<br />`scipy.interpolate.lagrange()` 求系数<br />`scipy.interpolate.interp1d()` cubic:三次 |
|      7.1.2       |         二维插值         |                网格节点插值<br />散乱数据插值                | `scipy.interpolate.interp2d()`<br />`scipy.interpolate.interpn()` |
|      7.1.3       |       Python 插值        |  从散乱数据点估计原函数<br />进而求积分、微分等解决实际问题  | `scipy.interpolate.UnivariateSpline()`<br />`(scipy)[Spline].integral()`<br />`np.trapz()` 梯形面积积分<br />`(scipy)[Spline].derivative()`<br />`scipy.interpolate.griddata()` 散乱插值 |
|      7.2.2       |     线性最小二乘拟合     |                拟合函数是一个函数系的线性组合                | `A = np.linalg.pinv(R) @ Y`<br />`ax.contour()` 绘制椭圆<br />`np.polyfit()` 返回拟合多项式的系数 |
|      7.2.3       |    非线性最小二乘拟合    |               拟合函数不能视为函数系的线性组合               | `popt = curve_fit(f, x0, y0)[0]`<br />`least_squares(err, x0, args=(..))` |
|       7.3        |         函数逼近         |                    用简单函数逼近复杂函数                    |                           `sympy`                            |
|       7.4        |            -             |                   ▲ 黄河小浪底调水调沙问题                   |       三次样条插值<br />根据剩余标准差确定拟合函数次数       |
|       8.1        |      常微分方程问题      |               ● 物体冷却模型<br />目标跟踪问题               |                         牛顿冷却定律                         |
|       8.2        |      传染病预测问题      |        指数传播模型<br />SI 模型、SIS 模型、SIR 模型         |                              -                               |
|       8.3        |     常微分方程的求解     | 符号解 (解析解)<br />数值解<br />● 洛伦兹模型 (自洽常微分方程 )<br />● 狗追人模型 (目标跟踪问题) | `sp.dsolve(eq, func, ics)`<br />`sp.lambdify()`<br />`scipy.integrate.odient()` |

