import pandas as pd
import numpy as np


def _iv_for_grouping(categorical_series, target):
    assert target.isnull().sum()==0.0, "Missing in targets!"
    df = pd.DataFrame({'tr':target,
                       'bucket':categorical_series,
                       'cnt': 1

                       })
    all_tr = target.mean()
    all_cnt = df.shape[0]

    return (df.groupby('bucket')
            .agg({'tr': 'mean', 'cnt': 'sum'})
            .assign(tr=lambda x: np.clip(x['tr'], 0.001, 0.999))
            .eval('     (    (tr/{all_tr}) -    ((1-tr)/(1-{all_tr})) )'
                  '   * ( log(tr/{all_tr}) - log((1-tr)/(1-{all_tr})) )'
                  '   * ( cnt/{all_cnt})'
                  ''.format(all_tr=all_tr, all_cnt=all_cnt))
            .sum())