import time
import numpy
import pandas
import scipy
import sklearn
from sklearn import metrics
import os
import multiprocessing

this_py_file = os.path.dirname(os.path.realpath(__file__))


data = pandas.read_csv(this_py_file+'/evaluation/MFTE_Python_Eval_Results_filtered.csv', keep_default_na=False)

print(data)

print(data.dtypes)

data['TagGold'].value_counts()

min_n = 100
#exclude_tags = ['NONE', 'UNCLEAR', 'none', 'unclear']

data['Count'] = data.groupby('TagGold')['TagGold'].transform(len)
enough_data = data['Count'] >= min_n
#valid_tag_gold = ~data['TagGold'].isin(exclude_tags)
#valid_tag = ~data['Tag'].isin(exclude_tags)
#data_filtered = data[enough_data & valid_tag_gold & valid_tag]
data_filtered = data[enough_data]

data_filtered['TagGold'].value_counts()

tags_remaining = set.union(set(data_filtered['TagGold']), set(data_filtered['Tag']))

data_filtered['TagGold'] = pandas.Categorical(data_filtered['TagGold'], categories=tags_remaining)
data_filtered['Tag'] = pandas.Categorical(data_filtered['Tag'], categories=tags_remaining)

tags = data_filtered['TagGold'].unique()

precision, recall, f1, n = metrics.precision_recall_fscore_support(
 data['TagGold'],
 data['Tag'],
 labels=tags
)

results = pandas.DataFrame({
 'tag': tags,
 'precision': precision,
 'recall': recall,
 'f1': f1,
 'n': n
})

results

results = results.melt(id_vars=['tag', 'n'], var_name='metric')
# results['lower'] = numpy.nan
# results['upper'] = numpy.nan
# results['valid'] = False

results

n_resamples = 1000

data_bootstrap = data[data['TagGold'].isin(tags) | data['Tag'].isin(tags)]


def get_bootstrap_done (row: dict) -> dict:
    print(row['tag'], row['metric'], '... ', end='')
    start_time = time.time()
    temp = dict()
    if row['value'] == 1.0:
        print('skipping')
        temp['lower'] = numpy.nan
        temp['upper'] = numpy.nan
        temp['valid'] = False
        temp['tag'] = row['tag']
        temp['n'] = row['n']
    else:        
        if row['metric'] == 'precision':
            func = sklearn.metrics.precision_score
        elif row['metric'] == 'recall':
            func = sklearn.metrics.recall_score
        else:
            func = sklearn.metrics.f1_score
            
        def metric(y_true, y_pred):
            return func(y_true, y_pred, labels=[row['tag']], average=None)[0]

        boot = scipy.stats.bootstrap(
        (data_bootstrap['TagGold'], data_bootstrap['Tag']),
        metric,
        vectorized=False,
        paired=True,
        n_resamples=n_resamples,
        method='percentile',
        random_state=0
        )
        print('done', int(time.time() - start_time), 's')
        
        temp['lower'] = boot.confidence_interval.low
        temp['upper'] = boot.confidence_interval.high
        temp['valid'] = True
        temp['tag'] = row['tag']
        temp['metric'] = row['metric']
        temp['n'] = row['n']
    return temp

result_dict = results.T.to_dict().values()

if __name__ == "__main__":
    cpu_count = int(multiprocessing.cpu_count() / 2) #run half cpus
    with multiprocessing.Pool(cpu_count) as pool:
      result1 = pool.map(get_bootstrap_done, result_dict)
      result1 = pandas.DataFrame(result1)
      result1.to_csv(this_py_file+'/evaluation/Temp_CIs.csv', index=False)
      results = results.merge(result1, on=['tag', 'metric', 'n'], how='left')
      results.to_csv(this_py_file+'/evaluation/MFTE_Python_Eval_CIs.csv', index=False)



# for rownum, row in results.iterrows():
    
#     print(rownum, ':', row['tag'], row['metric'], '... ', end='')
#     start_time = time.time()
    
#     if row['value'] == 1.0:
#         print('skipping')
#         continue
    
#     if row['metric'] == 'precision':
#         func = sklearn.metrics.precision_score
#     elif row['metric'] == 'recall':
#         func = sklearn.metrics.recall_score
#     else:
#         func = sklearn.metrics.f1_score
        
#     def metric(y_true, y_pred):
#         return func(y_true, y_pred, labels=[row['tag']], average=None)[0]

#     boot = scipy.stats.bootstrap(
#      (data_bootstrap['TagGold'], data_bootstrap['Tag']),
#      metric,
#      vectorized=False,
#      paired=True,
#      n_resamples=n_resamples,
#      method='percentile',
#      random_state=0
#     )
    
#     print('done', int(time.time() - start_time), 's')
    
#     results.loc[rownum, 'lower'] = boot.confidence_interval.low
#     results.loc[rownum, 'upper'] = boot.confidence_interval.high
#     results.loc[rownum, 'valid'] = True
    
