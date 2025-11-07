import sys, os

from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath('..'))


import numpy as np

import pandas as pd

def _calculate_time_second(df):
    df = df.copy()
    df['game_time_seconds'] = pd.to_datetime(df['game_time']).astype('int64') // 1e9
    return df


def _calculate_period_second(df):
    df = df.copy()
    df['period_time_seconds'] = (
        df['period_time']
        .astype(str)
        .apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if ':' in x else np.nan)
    )
    df['previous_event_timeperiod_seconds'] = (
        df['previous_event_timeperiod']
        .astype(str)
        .apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if ':' in x else np.nan)
    )
    return df

def change_to_one_hot(df, cols):

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    encoded = ohe.fit_transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)


    df_encoded = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
    return df_encoded

