# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:23:17 2018
回归分析特征选择（包括Stepwise算法） python 实现
@author: acadsoc
"""
import scipy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.linear_model import  Lasso, LassoCV, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.formula import api as smf
import sys
import os

plt.style.use('ggplot') # 设置ggplot2画图风格
# 根据不同平台设置其中文字体路径
if sys.platform == 'linux':
    zh_font = matplotlib.font_manager.FontProperties(
        fname='/path/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/STZHONGS.TTF')
else:
    zh_font = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\STZHONGS.ttf')  # 设置中文字体

# 根据不同平台设定工作目录
if sys.platform == 'linux':
    os.chdir('path/jupyternb/ml/acadsoc/rollingRegression') # Linux path
else:
    os.chdir('D:/Python/rollingRegression') # Windows path

class featureSelection():
    '''
    多元线性回归特征选择类。
    
    参数
    ----
    random_state : int，默认是None
        随机种子。
        
    属性
    ----
    elasticnet_rs_best : model
        弹性网络随机搜索最佳模型。
    elasticnet_rs_feat_selected_ : dataframe
        弹性网络随机搜索最佳模型选择的系数大于0的变量。
    elasticnet_rs_R2_ : float
        弹性网络随机搜索最佳模型Rsquared。
    eln : model
        弹性网络。
    elasticnet_coef_ : dataframe
        弹性网络系数。
    elasticnet_feat_selected_ : list
        弹性网络选择系数大于0的变量。
    elasticnet_feat_ : float
        弹性网络Rsquared。
    rf_rs_best : model
        随机森林随机搜索最佳模型。
    rf_rs_feat_impo_ : dataframe
        随机森林随机搜索变量重要性排序。
    rf_rs_feat_selected_ : list
        随机森林随机搜索累积重要性大于impo_cum_threshold的变量列表。
    rf_rs_R2_ : float
        随机森林随机搜索Rsquared。
    rf_feat_impo_ : dataframe
        随机森林变量重要性排序。
    rf_feat_selected_ : list
        随机森林累积重要性大于impo_cum_threshold的变量列表。
    rf_R2_ : float
        随机森林Rsquared。
    stepwise_model : model
        逐步回归模型。
    '''      
    def __init__(self, random_state=None):
        self.random_state = random_state # 随机种子        

    def elasticNetRandomSearch(self, df, cv=10, n_iter=1000, n_jobs=-1, intercept=True,
                               normalize=True):
        '''
        ElasticNet随机搜索，搜索最佳模型。
        
        参数
        ----
        df : dataframe
            分析用数据框，response为第一列。
        cv : int, 默认是10
            交叉验证次数。
        n_iter : int, 默认是1000
            最大迭代次数。
        n_jobs : int, 默认是-1
            使用cpu线程数，默认为-1，表示所有线程。
        intercept : bool, 默认是True
            是否有截距项。
        normalize : bool, 默认是True
            是否将数据标准化。  
        '''
        if normalize: # 如果需要标准化数据
            df_std = StandardScaler().fit_transform(df)
            df = pd.DataFrame(df_std, columns=df.columns, index=df.index)
            
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        eln = ElasticNet(fit_intercept=intercept)
        param_rs = {'alpha' : scipy.stats.expon(loc=0, scale=1),  # 模型需搜索的参数
                    'l1_ratio' : scipy.stats.uniform(loc=0, scale=1)}
        
        elasticnet_rs = RandomizedSearchCV(eln,  # 建立随机搜索
                                param_distributions=param_rs,
                                scoring='r2',
                                cv=cv,
                                n_iter=n_iter,
                                n_jobs=n_jobs)
        elasticnet_rs.fit(X, y)  # 模型训练        
        # 用最佳模型进行变量筛选变量、系数  
        self.elasticnet_rs_best = ElasticNet(alpha=elasticnet_rs.best_params_['alpha'],
                                          l1_ratio = elasticnet_rs.best_params_['l1_ratio'])
        self.elasticnet_rs_best.fit(X, y)
        coef = pd.DataFrame(self.elasticnet_rs_best.coef_, index=df.columns[1:],
                            columns=['系数']).sort_values(by='系数', axis=0, ascending=False)
        self.elasticnet_rs_feat_selected_ = coef[coef > 0].dropna(axis=1).columns
        self.elasticnet_rs_R2_ = 1 - np.mean((y.values.reshape(-1,1) -
                                              self.elasticnet_rs_best.predict(X).reshape(-1,1)) ** 2) / np.var(y)
        return self    
    
    def elasticNetFeatureSelectPlot(self, df, l1_ratio=.7, normalize=True, intercept=True,
                                    plot_width=12, plot_height=5, xlim_exp=[-5, 1], ylim=[-1, 1]):
        '''
        绘制ElasticNet正则化效果图。
        
        参数
        ----
        df : dataframe
            分析用数据框，response为第一列。
        l1_ratio : float, 默认是0.7
            l1正则化率。
        normalize : bool, 默认是True
            是否将数据标准化。  
        intercept : bool, 默认是True
            回归方程是否有常数项。
        plot_width : int, 默认是12
            画板宽度。
        plot_height : int, 默认是5
            画板高度。
        xlim_exp : list, 默认是[-5, 1]
            x轴显示指数取值范围。
        ylim : list, 默认是[-1, 1]
            y轴显示取值范围。
        '''
        if normalize: # 如果需要标准化数据
            df_std = StandardScaler().fit_transform(df)
            df = pd.DataFrame(df_std, columns=df.columns, index=df.index)  
        
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        
        plt.figure(figsize=(plot_width, plot_height))
        ax = plt.subplot(111)
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen',
                  'lightblue', 'gray', 'indigo', 'orange', 'seagreen', 'gold', 'purple']
        weights, params = [], []
        for alpha in np.arange(-5, 1, 0.1, dtype=float):
            eln = ElasticNet(alpha=10 ** alpha, l1_ratio=l1_ratio, random_state=123,
                             fit_intercept=intercept)
            eln.fit(X, y)
            weights.append(eln.coef_)
            params.append(10 ** alpha)
        
        weights = np.array(weights)
        for column, color in zip(range(weights.shape[1]), colors):
            plt.plot(params, weights[:, column], label=df.columns[column + 1], color=color)
        
        plt.axhline(0, color='black', linestyle='--', linewidth=3)
        plt.xlim(10 ** xlim_exp[0], 10 ** xlim_exp[1])
        plt.ylim(ylim)
        plt.title('弹性网络变量选择图', fontproperties=zh_font)
        plt.ylabel('权重系数', fontproperties=zh_font)
        plt.xlabel('$alpha$')
        plt.xscale('log')
        plt.xticks(10 ** np.arange(xlim_exp[0], xlim_exp[1], dtype=float),
                   10 ** np.arange(xlim_exp[0], xlim_exp[1], dtype=float))
        plt.legend(loc='best', prop=zh_font)
        ax.legend(prop=zh_font)
        #plt.grid()
        plt.show()
        return self
    
    ''''''
    def elasticNet(self, df, feat_selected=None, alpha=1, l1_ratio=.7, intercept=True, normalize=False):
        '''
        ElasticNet回归分析。
        
        参数
        ----
        df : dataframe
            分析用数据框，response为第一列。
        alpha : float, 默认是1
            alpha。
        l1_ratio : float, 默认是0.7
            l1正则化率。
        intercept : bool, 默认是True
            是否有截距项。
        normalize : bool, 默认是True
            是否将数据标准化。          
        '''
        if normalize: # 如果需要标准化数据
            df_std = StandardScaler().fit_transform(df)
            df = pd.DataFrame(df_std, columns=df.columns, index=df.index)  
        
        if feat_selected is not None:  # 如果输入了选择好的变量
            X = df[feat_selected]
        else:            
            X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        
        self.eln = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=intercept)
        self.eln.fit(X, y)  # 模型训练
        
        # 变量、系数，R2
        self.elasticnet_coef_ = pd.DataFrame(self.eln.coef_, index = X.columns,
                            columns=['系数']).sort_values(by='系数', ascending=False)
        self.elasticnet_feat_selected_ = self.elasticnet_coef_[self.elasticnet_coef_ > 0].dropna(axis=0).index
        self.elasticnet_R2_ = 1 - np.mean((y.values.reshape(-1,1) -
                                           self.eln.predict(X).reshape(-1,1)) ** 2) / np.var(y)
        return self        
    
    def featureBarhPlot(self, df_coef, figsize=(12, 6)):   
        '''
        画特征条形图（纵向排列）。
        
        参数
        ----
        df_coef : dataframe
            特征系数（重要性）数据框。
        fitsize : tuple, 默认是(12, 6)
            画布宽高。
        '''       
        coef = df_coef.sort_values(by=df_coef.columns[0], axis=0, ascending=True)
        plt.figure(figsize=figsize)
        y_label = np.arange(len(coef))
        plt.barh(y_label, coef.iloc[:, 0])
        plt.yticks(y_label, coef.index, fontproperties=zh_font)
        
        for i in np.arange(len(coef)):
            if coef.iloc[i, 0] >= 0:
                dist = 0.003 * coef.iloc[:, 0].max()
            else:
                dist = -0.02 * coef.iloc[:, 0].max()
            plt.text(coef.iloc[i, 0] + dist, i - 0.2, '%.3f' % coef.iloc[i, 0], fontproperties=zh_font)
            
        # plt.grid()
        plt.ylabel('特征', fontproperties=zh_font)
        plt.xlabel('特征系数', fontproperties=zh_font)
        plt.title('特征系数条形图', fontproperties=zh_font)
        plt.legend(prop=zh_font)
        plt.show()         
    
    def randomForestRandomSearch(self, df, cv=10, n_iter=100, n_jobs=-1, impo_cum_threshold=.85,
                                 normalize=True):
        '''
        RandomForest随机搜索，搜索最佳模型。
        
        参数
        ----
        df : dataframe
            分析用数据框，response为第一列。
        cv : int, 默认是10
            交叉验证次数。
        n_iter : int, 默认是100
            最大迭代次数。
        n_jobs : int, 默认是-1
            使用cpu线程数，默认为-1，表示所有线程。
        impo_cum_threshold : float, 默认是0.85
            按累积重要性选择变量阈值。
        normalize : bool, 默认是True
            是否将数据标准化。
        '''
        if normalize: # 如果需要标准化数据
            df_std = StandardScaler().fit_transform(df)
            df = pd.DataFrame(df_std, columns=df.columns, index=df.index)  
            
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        rf = RandomForestRegressor()
        param_rs = {'n_estimators' : np.arange(1, 500),  # 模型需搜索的参数
                    'max_features' : np.arange(1, X.shape[1] + 1)}
        
        rf_rs = RandomizedSearchCV(rf,  # 建立随机搜索
                                param_distributions=param_rs,
                                scoring='r2',
                                cv=cv,
                                n_iter=n_iter,
                                n_jobs=n_jobs)
        rf_rs.fit(X, y)  # 模型训练        
        # 用最佳模型进行变量筛选变量、系数  
        self.rf_rs_best = RandomForestRegressor(n_estimators=rf_rs.best_params_['n_estimators'],
                                          max_features=rf_rs.best_params_['max_features'])
        self.rf_rs_best.fit(X, y)
        self.rf_rs_feat_impo_ = pd.DataFrame(self.rf_rs_best.feature_importances_, index = df.columns[1:],
                            columns=['系数']).sort_values(by='系数', axis=0, ascending=False)
        
        n = 0
        for i, v in enumerate(self.rf_rs_feat_impo_.values.cumsum()):
            if v >= impo_cum_threshold:
                n = i
                break
                
        self.rf_rs_feat_selected_ = self.rf_rs_feat_impo_.index[:n+1]           
        self.rf_rs_R2_ = 1 - np.mean((y.values.reshape(-1,1) - \
                                              self.rf_rs_best.predict(X).reshape(-1,1)) ** 2) / np.var(y)
        return self
    
    def randomForest(self, df, feat_selected=None, impo_cum_threshold=.85,
                     n_estimators=100, max_features='auto', normalize=False):
        '''
        Randomforest回归分析。
        
        参数
        ----
        df : dataframe
            分析用数据框，response为第一列。
        feat_selected : list, 默认是None
            选择的特征。
        impo_cum_threshold : float, 默认是0.85
            按累积重要性选择变量阈值。
        n_estimators : int, 默认是100
            森林含树数。
        max_features : int, 默认是'auto'
            每课时最大选择特征数。        
        normalize : bool, 默认是True
            是否将数据标准化。
        '''    
        if normalize: # 如果需要标准化数据
            df_std = StandardScaler().fit_transform(df)
            df = pd.DataFrame(df_std, columns=df.columns, index=df.index)  
        
        if feat_selected is not None:  # 如果输入了选择好的变量
            X = df[feat_selected]
        else:            
            X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        
        self.rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features)
        self.rf.fit(X, y)  # 模型训练
        
        # 变量、系数，R2
        self.rf_feat_impo_ = pd.DataFrame(self.rf.feature_importances_, index = X.columns,
                            columns=['系数']).sort_values(by='系数', ascending=False)
        
        n = 0
        for i, v in enumerate(self.rf_feat_impo_.values.cumsum()):
            if v >= impo_cum_threshold:
                n = i
                break
                
        self.rf_feat_selected_ = self.rf_feat_impo_.index[:n+1]      
        self.rf_R2_ = 1 - np.mean((y.values.reshape(-1,1) - self.rf.predict(X).reshape(-1,1)) ** 2) / np.var(y)
        return self    
    
    def stepwise(self, df, response, intercept=True, normalize=False, criterion='bic',
                 f_pvalue_enter=.05, p_value_enter=.05, direction='backward', show_step=True,
                 criterion_enter=None, criterion_remove=None,max_iter=200, **kw):
                 
        '''
        逐步回归。
        
        参数
        ----
        df : dataframe
            分析用数据框，response为第一列。
        response : str
            回归分析相应变量。
        intercept : bool, 默认是True
            模型是否有截距项。
        criterion : str, 默认是'bic'
            逐步回归优化规则。
        f_pvalue_enter : float, 默认是.05
            当选择criterion=’ssr‘时，模型加入或移除变量的f_pvalue阈值。
        p_value_enter : float, 默认是.05
            当选择derection=’both‘时，移除变量的pvalue阈值。
        direction : str, 默认是'backward'
            逐步回归方向。
        show_step : bool, 默认是True
            是否显示逐步回归过程。
        criterion_enter : float, 默认是None
            当选择derection=’both‘或'forward'时，模型加入变量的相应的criterion阈值。
        criterion_remove : float, 默认是None
            当选择derection='backward'时，模型移除变量的相应的criterion阈值。
        max_iter : int, 默认是200
            模型最大迭代次数。
        '''
        criterion_list = ['bic', 'aic', 'ssr', 'rsquared', 'rsquared_adj']
        if criterion not in criterion_list:
            raise IOError('请输入正确的criterion, 必须是以下内容之一：', '\n', criterion_list)
            
        direction_list = ['backward', 'forward', 'both']
        if direction not in direction_list:
            raise IOError('请输入正确的direction, 必须是以下内容之一：', '\n', direction_list)
            
        # 默认p_enter参数    
        p_enter = {'bic':0.0, 'aic':0.0, 'ssr':0.05, 'rsquared':0.05, 'rsquared_adj':-0.05}
        if criterion_enter:  # 如果函数中对p_remove相应key传参，则变更该参数
            p_enter[criterion] = criterion_enter
            
        # 默认p_remove参数    
        p_remove = {'bic':0.01, 'aic':0.01, 'ssr':0.1, 'rsquared':0.05, 'rsquared_adj':-0.05}
        if criterion_remove:  # 如果函数中对p_remove相应key传参，则变更该参数
            p_remove[criterion] = criterion_remove
            
        if normalize: # 如果需要标准化数据
            intercept = False  # 截距强制设置为0
            df_std = StandardScaler().fit_transform(df)
            df = pd.DataFrame(df_std, columns=df.columns, index=df.index)  
                
        ''' forward '''
        if direction == 'forward':
            remaining = list(df.columns)  # 自变量集合
            remaining.remove(response)
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            if intercept: # 是否有截距
                formula = "{} ~ {} + 1".format(response, remaining[0])
            else:
                formula = "{} ~ {} - 1".format(response, remaining[0])
                    
            result = smf.ols(formula, df).fit() # 最小二乘法回归模型拟合            
            current_score = eval('result.' + criterion)
            best_new_score = eval('result.' + criterion)
                
            if show_step:    
                print('\nstepwise starting:\n')
            iter_times = 0
            # 当变量未剔除完，并且当前评分更新时进行循环
            while remaining and (current_score == best_new_score) and (iter_times<max_iter):
                scores_with_candidates = []  # 初始化变量以及其评分列表
                for candidate in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
                    if intercept: # 是否有截距
                        formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
                    else:
                        formula = "{} ~ {} - 1".format(response, ' + '.join(selected + [candidate]))
                        
                    result = smf.ols(formula, df).fit() # 最小二乘法回归模型拟合
                    fvalue = result.fvalue
                    f_pvalue = result.f_pvalue                    
                    score = eval('result.' + criterion)                    
                    scores_with_candidates.append((score, candidate, fvalue, f_pvalue)) # 记录此次循环的变量、评分列表
                    
                if criterion == 'ssr':  # 这几个指标取最小值进行优化
                    scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                    best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
                    if ((current_score - best_new_score) > p_enter[criterion]) and (best_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
                        remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                        selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                        current_score = best_new_score  # 更新当前评分
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
                                  (best_candidate, best_new_score, best_new_fvalue, best_new_f_pvalue))
                    elif (current_score - best_new_score) >= 0 and (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0: # 当评分差大于等于0，且为第一次迭代
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                        selected.append(remaining[0])
                        remaining.remove(remaining[0])
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
                elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                    scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                    best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
                    if (current_score - best_new_score) > p_enter[criterion]:  # 如果当前评分大于最新评分
                        remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                        selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                        current_score = best_new_score  # 更新当前评分
                        #print(iter_times)
                        if show_step:  # 是否显示逐步回归过程  
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif (current_score - best_new_score) >= 0 and iter_times == 0: # 当评分差大于等于0，且为第一次迭代
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                        selected.append(remaining[0])
                        remaining.remove(remaining[0])
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
                else:
                    scores_with_candidates.sort()
                    best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()
                    if (best_new_score - current_score) > p_enter[criterion]:
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        print(iter_times, flush=True)
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif (best_new_score - current_score) >= 0 and iter_times == 0: # 当评分差大于等于0，且为第一次迭代
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                        selected.append(remaining[0])
                        remaining.remove(remaining[0])
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
                iter_times += 1                        

            if intercept: # 是否有截距
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
            else:
                formula = "{} ~ {} - 1".format(response, ' + '.join(selected))
                
            self.stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合
            
            if show_step:  # 是否显示逐步回归过程                
                print('\nLinear regression model:', '\n  ', self.stepwise_model.model.formula)
                print('\n', self.stepwise_model.summary())
        
        ''' backward '''
        if direction == 'backward':
            remaining, selected = set(df.columns), set(df.columns)  # 自变量集合
            remaining.remove(response)
            selected.remove(response)  # 初始化选入模型的变量列表
             # 初始化当前评分,最优新评分
            if intercept: # 是否有截距
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
            else:
                formula = "{} ~ {} - 1".format(response, ' + '.join(selected))
                    
            result = smf.ols(formula, df).fit() # 最小二乘法回归模型拟合            
            current_score = eval('result.' + criterion)
            worst_new_score = eval('result.' + criterion)
                
            if show_step:    
                print('\nstepwise starting:\n')
            iter_times = 0
            # 当变量未剔除完，并且当前评分更新时进行循环
            while remaining and (current_score == worst_new_score) and (iter_times<max_iter):
                scores_with_eliminations = []  # 初始化变量以及其评分列表
                for elimination in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
                    if intercept: # 是否有截距
                        formula = "{} ~ {} + 1".format(response, ' + '.join(selected - set(elimination)))
                    else:
                        formula = "{} ~ {} - 1".format(response, ' + '.join(selected - set(elimination)))
                        
                    result = smf.ols(formula, df).fit() # 最小二乘法回归模型拟合
                    fvalue = result.fvalue
                    f_pvalue = result.f_pvalue                    
                    score = eval('result.' + criterion)                    
                    scores_with_eliminations.append((score, elimination, fvalue, f_pvalue)) # 记录此次循环的变量、评分列表
                    
                if criterion == 'ssr':  # 这几个指标取最小值进行优化
                    scores_with_eliminations.sort(reverse=False)  # 对评分列表进行降序排序
                    worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()  # 提取最小分数及其对应变量
                    if ((worst_new_score - current_score) < p_remove[criterion]) and (worst_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
                        remaining.remove(worst_elimination)  # 从剩余未评分变量中剔除最新最优分对应的变量
                        selected.remove(worst_elimination)  # 从已选变量列表中剔除最新最优分对应的变量
                        current_score = worst_new_score  # 更新当前评分
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Removing %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
                                  (worst_elimination, worst_new_score, worst_new_fvalue, worst_new_f_pvalue))
                elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                    scores_with_eliminations.sort(reverse=False)  # 对评分列表进行降序排序
                    worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()  # 提取最小分数及其对应变量
                    if (worst_new_score - current_score) < p_remove[criterion]:  # 如果评分变动不显著
                        remaining.remove(worst_elimination)  # 从剩余未评分变量中剔除最新最优分对应的变量
                        selected.remove(worst_elimination)  # 从已选变量列表中剔除最新最优分对应的变量
                        current_score = worst_new_score  # 更新当前评分
                        if show_step:  # 是否显示逐步回归过程  
                            print('Removing %s, %s = %.3f' % (worst_elimination, criterion, worst_new_score))                        
                else:
                    scores_with_eliminations.sort(reverse=True)
                    worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()
                    if (current_score - worst_new_score) < p_remove[criterion]:
                        remaining.remove(worst_elimination)
                        selected.remove(worst_elimination)
                        current_score = worst_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Removing %s, %s = %.3f' % (worst_elimination, criterion, worst_new_score))                    
                iter_times += 1
                
            if intercept: # 是否有截距
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
            else:
                formula = "{} ~ {} - 1".format(response, ' + '.join(selected))
                
            self.stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合
            
            if show_step:  # 是否显示逐步回归过程                
                print('\nLinear regression model:', '\n  ', self.stepwise_model.model.formula)
                print('\n', self.stepwise_model.summary())
        
        ''' both '''
        if direction == 'both':
            remaining = list(df.columns)  # 自变量集合
            remaining.remove(response)
            selected = []  # 初始化选入模型的变量列表
            # 初始化当前评分,最优新评分
            if intercept: # 是否有截距
                formula = "{} ~ {} + 1".format(response, remaining[0])
            else:
                formula = "{} ~ {} - 1".format(response, remaining[0])
                    
            result = smf.ols(formula, df).fit() # 最小二乘法回归模型拟合            
            current_score = eval('result.' + criterion)
            best_new_score = eval('result.' + criterion)
                
            if show_step:    
                print('\nstepwise starting:\n')
            # 当变量未剔除完，并且当前评分更新时进行循环
            iter_times = 0
            while remaining and (current_score == best_new_score) and (iter_times<max_iter):
                scores_with_candidates = []  # 初始化变量以及其评分列表
                for candidate in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
                    if intercept: # 是否有截距
                        formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
                    else:
                        formula = "{} ~ {} - 1".format(response, ' + '.join(selected + [candidate]))
                        
                    result = smf.ols(formula, df).fit() # 最小二乘法回归模型拟合
                    fvalue = result.fvalue
                    f_pvalue = result.f_pvalue                    
                    score = eval('result.' + criterion)                    
                    scores_with_candidates.append((score, candidate, fvalue, f_pvalue)) # 记录此次循环的变量、评分列表
                    
                if criterion == 'ssr':  # 这几个指标取最小值进行优化
                    scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                    best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
                    if ((current_score - best_new_score) > p_enter[criterion]) and (best_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
                        remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                        selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                        current_score = best_new_score  # 更新当前评分
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
                                  (best_candidate, best_new_score, best_new_fvalue, best_new_f_pvalue))
                    elif (current_score - best_new_score) >= 0 and (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0: # 当评分差大于等于0，且为第一次迭代
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                        selected.append(remaining[0])
                        remaining.remove(remaining[0])
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
                elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                    scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                    best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
                    if (current_score - best_new_score) > p_enter[criterion]:  # 如果当前评分大于最新评分
                        remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                        selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                        current_score = best_new_score  # 更新当前评分
                        if show_step:  # 是否显示逐步回归过程  
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif (current_score - best_new_score) >= 0 and iter_times == 0: # 当评分差大于等于0，且为第一次迭代
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                        selected.append(remaining[0])
                        remaining.remove(remaining[0])
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
                else:
                    scores_with_candidates.sort()
                    best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()
                    if (best_new_score - current_score) > p_enter[criterion]:  # 当评分差大于p_enter
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif (best_new_score - current_score) >= 0 and iter_times == 0: # 当评分差大于等于0，且为第一次迭代
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                        selected.append(remaining[0])
                        remaining.remove(remaining[0])
                        if show_step:  # 是否显示逐步回归过程                             
                            print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
                            
                if intercept: # 是否有截距
                    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
                else:
                    formula = "{} ~ {} - 1".format(response, ' + '.join(selected))                    

                result = smf.ols(formula, df).fit()  # 最优模型拟合                    
                if iter_times >= 1: # 当第二次循环时判断变量的pvalue是否达标
                    if result.pvalues.max() > p_value_enter:
                        var_removed = result.pvalues[result.pvalues == result.pvalues.max()].index[0]
                        p_value_removed = result.pvalues[result.pvalues == result.pvalues.max()].values[0]
                        selected.remove(result.pvalues[result.pvalues == result.pvalues.max()].index[0])
                        if show_step:  # 是否显示逐步回归过程                
                            print('Removing %s, Pvalue = %.3f' % (var_removed, p_value_removed))
                iter_times += 1
                
            if intercept: # 是否有截距
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected))
            else:
                formula = "{} ~ {} - 1".format(response, ' + '.join(selected))
                
            self.stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合           
            if show_step:  # 是否显示逐步回归过程                
                print('\nLinear regression model:', '\n  ', self.stepwise_model.model.formula)
                print('\n', self.stepwise_model.summary())                
        # 最终模型选择的变量
        if intercept:
            self.stepwise_feat_selected_ = list(self.stepwise_model.params.index[1:])
        else:
            self.stepwise_feat_selected_ = list(self.stepwise_model.params.index)
        return self