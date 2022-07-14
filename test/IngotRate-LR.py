import numpy as np
import pandas as pd
import statsmodels.api as sm

file = r'/成锭率-Pandas.csv'
data = pd.read_csv(file)
# print(data)
data.columns = ['Ingot specifications',
                'Scrap input',
                'Return alloy steel',
                'Carbon steel',
                'Surface quality: scarring',
                'Surface quality: short length',
                'Surface quality: flake',
                'Surface quality: other',
                'Chemical composition: Mo',
                'Chemical composition: Ni',
                'Chemical composition: Cr',
                'Chemical composition: Mn',
                'Chemical composition: C',
                'Chemical composition: Si',
                'Ingot rate']


# # 生成自变量（除掉最后一列）
# x = sm.add_constant(data.iloc[:, 0:-1])
# print(x)
# # 生成因变量
# y = data['Ingot rate']
# print(y)
# # 生成模型
# model = sm.OLS(y, x)
# # 模型拟合
# result = model.fit()
# # 模型描述
# print(result.summary())

def looper(limit):
    # 先去掉了最无关的Carbon steel（碳素钢）
    cols = ['Ingot specifications',
            'Scrap input',
            'Return alloy steel',
            'Surface quality: scarring',
            'Surface quality: short length',
            'Surface quality: flake',
            'Surface quality: other',
            'Chemical composition: Mo',
            'Chemical composition: Ni',
            'Chemical composition: Cr',
            'Chemical composition: Mn',
            'Chemical composition: C',
            'Chemical composition: Si']
    for i in range(len(cols)):
        # 生成自变量
        x = sm.add_constant(data[cols])
        # 生成因变量
        y = data['Ingot rate']
        # 生成模型
        model = sm.OLS(y, x)
        # 模型拟合
        result = model.fit()
        # 得到结果中所有P值
        pvalues = result.pvalues
        # 把const列去掉
        pvalues.drop('const', inplace=True)
        # 选出最大的P值
        pmax = max(pvalues)
        if pmax > limit:
            # 删除P值最大的自变量
            ind = pvalues.idxmax()
            cols.remove(ind)
        else:
            return result


result = looper(0.8)
print(result.summary())
