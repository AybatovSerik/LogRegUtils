import numpy as np
from scipy.special import logit
from risksutils.visualization import _make_bucket
from sklearn.base import TransformerMixin, BaseEstimator


class Group:
    def __init__(
            self,
            target_num,
            non_target_num,
            target_num_all,
            non_target_num_all,
            border_left=None,
            border_right=None):
        self.target_num = target_num
        self.non_target_num = non_target_num
        self.target_num_all = target_num_all
        self.non_target_num_all = non_target_num_all
        self.border_left = border_left
        self.border_right = border_right

        self.target_rate = self.target_num/(self.target_num + self.non_target_num)
        self.target_rate_all = self.target_num_all/(self.target_num_all + self.non_target_num_all)

        self.N = self.target_num + self.non_target_num
        self.N_all = self.target_num_all + self.non_target_num_all
        self.target_rate, self.target_rate_all = np.clip([self.target_rate, self.target_rate_all], 0.001, 0.999)

    def woe(self):
        return logit(self.target_rate) - logit(self.target_rate_all)

    def iv(self):
        return self.woe()*(self.target_rate/self.target_rate_all - (1-self.target_rate)/(1-self.target_rate_all))*(self.N/self.N_all)

    def in_group(self,x):
        if (x>self.border_left) & (x<=self.border_right):
            return True
        else:
            return False

    def __add__(self, that):
        assert self.target_num_all == that.target_num_all, "Should be the same counts"
        assert self.non_target_num_all == that.non_target_num_all, "Should be the same counts"
        group_union = Group(
            self.target_num + that.target_num,
            self.non_target_num + that.non_target_num,
            self.target_num_all,
            self.non_target_num_all
        )
        group_union.in_group = lambda x: True if (self.in_group(x)|that.in_group(x)) else False
        return group_union


def sort_group_by_woe(groups):
    return sorted(groups, key=lambda x: x.woe())


def check_loss_iv_from_union(group1, group2, iv_threshold=0.001):
    group = group1 + group2
    return group1.iv() + group2.iv() - group.iv() < iv_threshold


def groups_union(groups, iv_threshold=0.001):
    new_groups = groups.copy()
    new_groups = sort_group_by_woe(new_groups)
    num_of_groups = len(groups)
    indexs = [i for i in range(num_of_groups)]
    indexs.reverse()
    for i in indexs[:-1]:
        group_1 = new_groups[i]
        group_2 = new_groups[i-1]
        if check_loss_iv_from_union(group_1, group_2, iv_threshold):
            group_union = group_1 + group_2
            new_groups.remove(group_1)
            new_groups.insert(i,group_union)
            new_groups.remove(group_2)
    return new_groups


def _aggregate_data_for_woe_grouping(df, feature, target, num_buck):
    df = df[[feature, target]].dropna()

    df_agg = (
        df.assign(bucket=lambda x: _make_bucket(x[feature], num_buck),
                  obj_count=1)
            .groupby('bucket', as_index=False)
            .agg({target: 'sum', 'obj_count': 'sum', feature: 'mean'})
            .dropna()
            .rename(columns={target: 'target_count'})
            .assign(obj_total=lambda x: x['obj_count'].sum(),
                    target_total=lambda x: x['target_count'].sum())
    )
    df_agg_borders = (df.assign(bucket=lambda x: _make_bucket(x[feature], num_buck),
                                obj_count=1)
                      .groupby('bucket', as_index=False)[feature]
                      .agg(['mean', 'min', 'max'])
                      )
    return df_agg, df_agg_borders


def make_groups_from_buckets(bucket_info, bucket_borders):
    groups = []
    num_of_groups = bucket_info.shape[0]
    left_border = -np.inf
    for i in range(num_of_groups):
        if i == 0:
            aggr_ = bucket_info.iloc[i, :]
            bord = bucket_borders.iloc[i, :]
            group = Group(
                aggr_['target_count'],
                aggr_['obj_count'] - aggr_['target_count'],
                aggr_['target_total'],
                aggr_['obj_total'] - aggr_['target_total'],
                left_border,
                bord['max']
            )
        elif i == (num_of_groups-1):
            aggr_ = bucket_info.iloc[i, :]
            bord = bucket_borders.iloc[i, :]
            group = Group(
                aggr_['target_count'],
                aggr_['obj_count'] - aggr_['target_count'],
                aggr_['target_total'],
                aggr_['obj_total'] - aggr_['target_total'],
                left_border,
                np.inf
            )
        else:
            aggr_ = bucket_info.iloc[i, :]
            bord = bucket_borders.iloc[i, :]
            group = Group(
                aggr_['target_count'],
                aggr_['obj_count'] - aggr_['target_count'],
                aggr_['target_total'],
                aggr_['obj_total'] - aggr_['target_total'],
                left_border,
                bord['max']
            )

        left_border = bord['max']
        groups.append(group)

    return groups


def woe_from_groups(x, groups):
    for group in groups:
        if group.in_group(x):
            return group.woe()
    print(f"Warning, do not found any group for value {x}")


class Grouping(TransformerMixin, BaseEstimator):
    def __init__(self, iv_threshold=0.001):
        self.iv_threshold = iv_threshold

    def _reset(self):
        if hasattr(self, 'groups'):
            del self.groups_dict

    def fit_one_feature(self, df, feature, target, num_bucket):
        bucket_info, bucket_borders = _aggregate_data_for_woe_grouping(df, feature, target, num_bucket)
        groups = make_groups_from_buckets(bucket_info, bucket_borders)
        groups = groups_union(groups, self.iv_threshold)
        return groups

    def fit(self, df, features, target, num_bucket=20):
        self._reset()
        if type(features) == str:
            features = [features]

        self.groups_dict = dict()

        for feature in features:
            self.groups_dict[feature] = self.fit_one_feature(df, feature, target, num_bucket)

    def transform_one_feature(self, df, feature):
        groups = self.groups_dict[feature]
        return df[feature].apply(lambda x: woe_from_groups(x,groups))

    def transform(self, df, features):
        df_transform = df.copy()
        for feature in features:
            df_transform[feature + '_woe'] = self.transform_one_feature(df, feature)
        return df_transform
