|       索引       |                名称                 |                             概述                             |                           核心代码                           |
| :--------------: | :---------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      3.1.1       |              差分方程               |                        求递推数列通项                        |                    `sp.rsolve()` 直接求解                    |
|      3.1.2       |      莱斯利 (Leslie) 种群模型       |        已知初始状态、生育率、存活率<br/>预测种群总量         | $\small \boldsymbol A = \boldsymbol {PD} \boldsymbol P^{-1}$<br/>`P, D = sp.Matrix(ins=A).diagonalize()`<br/>特征值分解\|相似对角化<br/>matplotlib 刻度与刻度标签控制 |
|      3.1.3       |            PageRank 算法            |               图论、马尔可夫链、互达的概率排序               |             有向图构造<br/>`ax.bar()` 柱状图绘制             |
|  $\rightarrow$   |            随机冲浪模型             |                     非互达，引入阻尼因子                     |                              -                               |
|      3.2.2       |            推荐系统评分             |                 基于皮尔逊相关系数的评分预测                 | `np.corrcoef()` 按行返回相关系数矩阵<br/>`ax.imshow()` 热力图绘制 |
|  $\rightarrow$   |       基于奇异值分解压缩数据        |                 稀疏矩阵降维<br/>余弦相似度                  | $\small \boldsymbol A = \boldsymbol U\begin{bmatrix}\boldsymbol \Sigma 	&\boldsymbol 0\\ \boldsymbol 0		&\boldsymbol 0\end{bmatrix}\boldsymbol V^{\rm T}$<br/>`U, S, VT = np.linalg.svd(A)`<br/>其中 `np.diag(S)`=$\small \begin{bmatrix}\boldsymbol \Sigma 	&\boldsymbol 0\\ \boldsymbol 0		&\boldsymbol 0\end{bmatrix}$<br/>列压缩数据降维范式 |
|      3.2.2       |         利用SVD进行图像压缩         |                    只保留那些较大的奇异值                    | 数字图像处理库 PIL.Image<br/>妙用`ax.imshow(cmap='gray')`绘制灰度图<br/>对矩阵整体进行数据压缩范式 |
|      4.1.2       |          线性规划模型 (LP)          | ● 企业安排生产问题<br/>● 项目投资问题<br/>● 仓库租借问题<br/>● 最小费用运输问题 | `import cvxpy as cp`<br/>`x = cp.Variable()`<br/>`obj = cp.Maximize[Minimize]()`<br/>`cons = [...]`<br/>`prob = cp.Problem(obj, cons)`<br>`prob.solve(solver='...')` |
|      4.2.1       |           0-1整数规划模型           |           背包问题、指派问题<br />旅行商问题 (TSP)           |          *指派问题代码见 4h5*<br />*TSP 代码见 4.4*          |
|      4.2.2       |            整数规划模型             | ● 工时安排问题<br/>● 装修分配任务问题 (非标准指派问题)<br/>● 网点覆盖问题 (双决策变量) | `sklearn.metrics.euclidean_distances`<br/>两个矩阵的成对平方欧氏距离 |
|       4.3        |           多目标规划模型            | ▲ 组合投资问题：<br/>将可供投资的资金分成 $n\!+\!1$ 份，<br/>分别购买 $n\!+\!1$ 种资产，<br/>同时兼顾投资的净收益和风险<br /> | 多目标模型的目标函数线性化<br />[优化模型线性化方法总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/361766549)<br />引入变量$x_{n+1}={\rm max} \left\{q_ix_i\right\}$<br />投资风险为总收益的方差 |
|       4.4        |             旅行商模型              |         ▲ 比赛项目排序问题<br />引入虚拟项目构成闭环         |           `NaN`的处理：`data[np.isnan(data)] = 0`            |
|        4h        |                  -                  | ● 限制运量的最小费用运输问题<br />● 面试顺序问题<br />● 带约束的背包问题<br />● 钢管下料问题 |                              -                               |
|       5.2        |           非线性规划模型            |                        ▲ 彩电生产问题                        | 灵敏性分析相关计算和表述<br />`sympy`模块`diff`，`subs`等函数的使用<br />三维图坐标刻度控制 |
|       5.3        |            二次规划模型             | 目标函数为决策向量的二次函数<br />约束条件均为线性的<br />▲ 投资组合 (portfolio) 问题 | `cp.quad_form(x, c)` 返回二次型<br />`np.cov()` 按行返回协方差矩阵 |
|       5.4        |            非凸规划模型             |                              -                               | `scipy.optimize.minimize()`局部最优解<br />*先用 `cvxpy` 求解，发现非凸后再用 `minimize`<br />*另外，对于无约束问题，可以采用<br />`scipy.optimize.basinhopping()`<br />获得全局最优解 |
|  $\rightarrow$   |                  -                  |                       ● 供应与选址问题                       |             `minimize` 多元决策变量的划分与解包              |
|       5.5        |           多目标规划模型            |                       ▲ 生产与污染问题                       |                              -                               |
|       5.6        |                  -                  |                        ▲ 飞行管理问题                        | 绘制箭头 `arrow()`<br />相对运动、三角学<br />`numpy` 模块中的复数、辐角 `angle()`<br />Python特性：延迟绑定 |
|       5h6        |                  -                  |                     ● 组合投资问题<br />                     |                              -                               |
|       5h7        |                  -                  |                        ● 生产计划问题                        |                        `cp.cumsum()`                         |
| 6.3.1<br />6.3.2 |             最短路算法              |                  Dijkstra 算法、Floyd 算法                   | `nx.dijkstra_path()`<br />`nx.dijkstra_path_length()`<br />`nx.shortest_path()`<br />`nx.shorted_path_length()`<br />`nx.floyd_warshall_numpy()` |
|      6.3.3       |             最短路应用              |                     ● 设备更新问题<br />                     |                     `nx.shortest_path()`                     |
|      6.3.3       |             最短路应用              |                          ● 选址问题                          |                 `np.argmin()`，`np.argmax()`                 |
|      6.3.4       |                  -                  |                  最短路问题的 0-1 规划模型                   |                              -                               |
|      6.4.1       |             最小生成树              |                        ● 架设电线问题                        |                 `nx.minimum_spanning_tree()`                 |
|      6.4.2       |                  -                  |                最小生成树问题的 0-1 规划模型                 |                              -                               |
|       6.5        |              着色问题               | 物资储存问题<br />无线交换设备的波长分配问题<br />● 会议安排问题 |                  计算色数的整数线性规划模型                  |
|      6.6.1       |             最大流问题              |       最大流问题的 0-1 规划模型<br />● 多对多招聘问题        |      `nx.maximum_flow()`<br />从字典中导出邻接矩阵范式       |
|      6..6.2      |           最小费用流问题            |              ● 运费网络模型<br />最小费用最大流              |      `nx.max_flow_min_cost()`<br />`nx.cost_of_flow()`       |
|      6.7.4       |         计划网络的优化问题          |                          ● 赶工问题                          |                              -                               |
|      6.7.5       |                  -                  |               求完成作业的期望和实现事件的概率               |   `scipy.stats.norm` (正态分布)<br />中的 `pdf()`，`cdf()`   |
|       6.8        |                  -                  |                     ▲ 钢管订购和运输问题                     |                 `nx.floyd_warshall_numpy()`                  |
|       6h4        |                  -                  |                   ● 允许售出的设备更新问题                   |                     `nx.shortest_path()`                     |
|       6h7        |                  -                  |                      无向图的度量图绘制                      |                  `ax.plot()` 可绘制顶点和边                  |
|      7.1.1       |              一维插值               | 多项式插值<br />拉格朗日插值法<br />分段线性插值、三次样条插值 | `np.vander()` 返回范德蒙行列式<br />`scipy.interpolate.lagrange()` 求系数<br />`scipy.interpolate.interp1d()` cubic:三次 |
|      7.1.2       |              二维插值               |                网格节点插值<br />散乱数据插值                | `scipy.interpolate.interp2d()`<br />`scipy.interpolate.interpn()` |
|      7.1.3       |             Python 插值             | 从散乱数据点估计原函数<br />进而求积分、微分等以解决实际问题 | `scipy.interpolate.UnivariateSpline()`<br />`(scipy)[Spline].integral()`<br />`np.trapz()` 梯形面积积分<br />`(scipy)[Spline].derivative()`<br />`scipy.interpolate.griddata()` 散乱插值 |
|      7.2.2       |          线性最小二乘拟合           |                拟合函数是一个函数系的线性组合                | `A = np.linalg.pinv(R) @ Y`<br />`ax.contour()` 绘制椭圆<br />`np.polyfit()` 返回拟合多项式的系数 |
|      7.2.3       |         非线性最小二乘拟合          |               拟合函数不能视为函数系的线性组合               | `popt = curve_fit(f, x0, y0)[0]`<br />`least_squares(err, x0, args=(..))` |
|       7.3        |              函数逼近               |                    用简单函数逼近复杂函数                    |                           `sympy`                            |
|       7.4        |                  -                  |                   ▲ 黄河小浪底调水调沙问题                   |       三次样条插值<br />根据剩余标准差确定拟合函数次数       |
|       8.1        |           常微分方程问题            |               ● 物体冷却模型<br />目标跟踪问题               |                         牛顿冷却定律                         |
|       8.2        |           传染病预测问题            |        指数传播模型<br />SI 模型、SIS 模型、SIR 模型         |                              -                               |
|       8.3        |       常微分方程 (组) 的求解        | 符号解 (解析解)<br />数值解<br />● 洛伦兹模型 (自洽常微分方程 )<br />● 狗追人模型 (目标跟踪问题) | `sp.dsolve(eq, func, ics)`<br />`sp.lambdify()`<br />`scipy.integrate.odient()` |
|      8.4.1       |            Malthus 模型             |                          无限制增长                          |                              -                               |
|      8.4.2       |            Logistic 模型            |         增长率为人口的减函数<br />向前差分，向后差分         |                         `np.diff()`                          |
|      8.4.3       |            种群竞争模型             |                              -                               |                              -                               |
|      8.4.3       |         捕食者-被捕食者模型         |                   弱肉强食问题 (差分方程)                    |               `ax.quiver(x, y, u, v)` 矢量场图               |
|       8.5        |            差分方程建模             |              离散化<br />差分方程的解及其稳定性              |                              -                               |
|       8.6        |          差分方程+搜索算法          |                        ▲ 最优捕鱼策略                        |                 符号运算解方程过慢的解决方法                 |
|       8h10       |            微分方程问题             |                        ▲ 药物中毒问题                        |                              -                               |
|       8h11       |            差分方程建模             |                         ● 预测销售量                         |                              -                               |
|       9.1        |           简单的统计分析            |                        `scipy.stats`                         |                      详见 Jupyter 文件                       |
|      9.2.2       |             统计量计算              |                       NumPy \| Pandas                        |                              -                               |
|      9.2.3       |             统计图绘制              |       直方图<br />箱线图<br />经验分布函数<br />Q-Q 图       | `ax.hist()`<br />`ax.boxplot()`<br />`ax.hist(cumulative=True)`<br />- |
|      9.3.1       |              参数估计               |                      标准误差 $\rm SEM$                      |                     `scipy.stats.sem()`                      |
|      9.3.2       |            参数假设检验             | $Z$ 检验<br />$t$ 检验<br />两个正态总体均值差的 $t$ 检验<br />$\chi^2$ 检验<br />柯尔莫哥洛夫检验 | `statsmodels.stats.weightstats.ztest()`<br />`scipy.stats.ttest_1samp()`<br />`statsmodels.stats.weightstats.ttest_ind()`<br />`scipy.stats.chisquare()`<br />`scipy.stats.kstest()` |
|      9.4.1       |         单因素方差分析方法          |               判段单个因素对指标是否有显著影响               |              `statsmodels.api.stats.anova_lm()`              |
|      9.4.2       |         双因素方差分析方法          | 无交互影响的双因素方差分析<br />关于交互效应的双因素方差分析 |                             同上                             |
|       9h6        |   非参数假设检验<br />模拟与搜索    |                       ● 自动化车床管理                       |                              -                               |
|       10.1       |            一元线性回归             |                     野值检验<br />误差棒                     | `sm.formula.ols('y~x', mod_dic)`<br />`mod.summary()`<br />`mod.outlier_test()`<br />`ax.errorbar(num, mod.resid, r, fmt=)` |
|       10.2       |            多元线性回归             |                              -                               |   `sm.formula.ols('y~x1+x2', mod_dic)`<br />`sm.OLS(y, X)`   |
|       10.3       |             多项式回归              |         纯二次、交叉二次、混合二次<br />回归系数检验         |                              -                               |
|       10.4       |              逐步回归               |                从众多变量中有效地挑选重要变量                |                              -                               |
|      10.5.1      |  分组数据的<br />Logistic 回归模型  | 把自变量按一定值分组，<br />构建 Logistic 模型预测因变量（概率） | 线性化（逻辑变换）后 `sm.OLS()`<br />直接 `sm.formula.glm('y~x', [...])` |
|      10.5.2      | 未分组数据的<br />Logistic 回归模型 | 构建关于多个自变量的 Logistic模型，<br />对因变量的行为 (0-1) 进行预测 | `sm.formula.logit('y~x1+[...], mod_dic')`<br />`sm.formula.glm('y~x1+[...]', [...])` |
|      10.5.3      |           Probit 回归模型           |                   假设随机扰动服从正态分布                   | 分组：Probit 变换后`sm.OLS()`<br />未分组：`sm.formula.probit()` |
|      10.5.4      |            赔率和赔率比             |                   Logistic 模型参数的解释                    |                              -                               |
|      11.1.4      |              系统聚类               | 最短距离法<br />最长距离法<br />重心法、类平均法、离差平方和法 | `sch.linkage()`<br />`sch.fcluster()`<br />`sch.dendrogram()`<br />计算距离 `scipy.spatial.distance.pdist()`<br />标准化（无量纲化） `scipy.stats.zscore()` |
|      11.1.5      |              动态聚类               |    $K$ 均值聚类<br />$K$ 均值聚类法最佳簇数 $k$ 值的确定     | `sklearn.cluster.KMeans(k).fit(data)`<br />`sklearn.metrics.silhouette_score()` |
|      11.1.6      |             R 型聚类法              |                        ● 服装标准制定                        |                       `sch.linkage()`                        |
|      11.2.1      |             距离判别法              |                              -                               |         `scipy.spatial.distance.mahalanobis()`<br />         |
|      11.2.2      |             Fisher 判别             |                              -                               |                              -                               |
|      11.2.3      |           判别标准的评价            |                  回代误判率<br />交叉误判率                  |         `sklearn.model_selection.cross_val_score()`          |
|       11h5       |       距离判别和 Fisher 判别        |                       判断样本所属种类                       |                       `LDA()`，`KNC()`                       |
|      12.1.1      |             主成分分析              |             数据降维<br />去除变量的线性相关信息             |                `sklearn.decomposition.PCA()`                 |
|      12.1.2      |           主成分回归分析            |                  降维，用主成分拟合回归方程                  |                              -                               |
|      12.1.3      |            核主成分分析             |            非线性相关问题<br />● 高校科创能力评价            |             `sklearn.decomposition.KernelPCA()`              |
|      12.2.2      |              因子分析               |                        ● 学生综合评价                        |              `factor_analyzer.FactorAnalyzer()`              |
|       13.3       |         偏最小二乘回归分析          | 多对多回归分析<br />▲ 体能训练数据回归建模<br />● 交通业和旅游业的回归分析 |          `skl.cross_decomposition.PLSRegression()`           |
|       14.2       |             数据预处理              |      一致化<br />规范化（无量纲化）<br />定性指标定量化      |                   `sklearn.preprocessing`                    |
|       14.3       |          综合评价数学模型           | *past: 主成分分析法，PageRank 法*<br />线性加权综合评价法<br />TOPSIS 法（理想解法）<br />灰色关联度分析<br />熵值法、秩和比法 |                `scipy.stats.rankdata()` 编秩                 |
|       14.4       |            模糊综合评价             |      处理具有模糊性的评价因素<br />或评价对象，非常好用      |                              -                               |
|       14.5       |            数据包络分析             | CCR模型 (C^2^R)<br />适用于具有多输入多输出的系统<br />● 可持续发展评价 |                              -                               |
|       14.6       |          模糊综合评价实例           |                       ▲ 招聘公务员问题                       |                              -                               |
|       14h4       |         模糊聚类 (FCM) 分析         |                  某样本以一定比率隶属于某簇                  |     `fcmeans.FCM()`<br />聚类验证图的绘制（只能有两个）      |
|      15.1.1      |            GM(1,1) 模型             | 只适用于具有较强指数规律的序列<br />只能描述单调变化过程<br />对于非单调的振荡数据则不适用 |                      `def GM1_1(x0, m)`                      |
|  $\rightarrow$   |              误差检验               |                      相对误差和级比误差                      |   `def rela_err(x0, x0hat)`<br />`def ratio_err(x0, uhat)`   |
|      15.1.2      |            GM(2,1) 模型             |                       可以用于振荡数据                       |                      `def GM2_1(x0, m)`                      |
|      15.1.2      |            DGM(2,1) 模型            |          针对GM的稳定性不足、<br />误差较大做的改进          |                     `def DGM2_1(x0, m)`                      |
|      15.1.2      |            Verhulst 模型            | 主要用来描述具有饱和状态的过程，<br />即 S 形过程，常用于<br />人口、经济、繁殖等的预测 |                    `def Verhulst(x0, m)`                     |
|       15.2       |            马尔可夫预测             | 系统未来时刻的情况只与现在有关<br />而与过去的历史无直接关系、<br />马氏链的极限分布 |                              -                               |
|      15.3.2      |               感知器                |                      对输入向量进行分类                      |             `sklearn.linear_model.Perceptron()`              |
|      15.3.3      |          多层感知机分类器           |                             分类                             |           `sklearn.neural_network.MLPClassifier()`           |
|      15.3.3      |         多层感知机回归分析          |                           回归分析                           |           `sklearn.neural_network.MLPRegressor()`            |
|       16.3       |         零和博弈的混合策略          |                   方程组解法和线性规划解法                   |                              -                               |
|       16.4       |         非合作的双矩阵博弈          |                     纯策略解和混合策略解                     |           scipy 的 `optimize()` 约束条件的注意事项           |
|       16h4       |    0-1 整数规划<br />满意度评价     |                        ● 相亲配对问题                        |                              -                               |
|       17.1       |        偏微分方程的定解问题         |                    椭圆型、抛物型、双曲型                    |                              -                               |
|       17.2       |       简单偏微分方程的符号解        |                              -                               |                        `sp.pdsolve()`                        |

