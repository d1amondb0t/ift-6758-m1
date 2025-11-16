import sys, os

from sklearn.preprocessing import OneHotEncoder
import wandb
import joblib
from dotenv import load_dotenv
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

def change_to_one_hot(df, cols, ohe):
    if ohe is None:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
        encoded = ohe.fit_transform(df[cols]) #Just doing a fit transform here
    else:
        encoded = ohe.transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
    df_encoded = pd.concat([df.drop(columns=cols), encoded_df], axis=1) 
    return df_encoded, ohe

def save_encoders_to_wanddb(one_hot_encoder, scalar, project='IFT6758-2025-B01') -> None:
    '''
    Saves the encoders to wanddb, we use this in case we need to perform the same encoding on our test sets
    So we do not have to replicate the splits to figure out the econding
    '''
    joblib.dump(one_hot_encoder, "one_hot_encoder.pkl")
    joblib.dump(scalar, "scalar.pkl")     
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    run = wandb.init(project=project)
    artifact = wandb.Artifact(
        name="preprocessing",
        type="preprocessor"
    )
    artifact.add_file("one_hot_encoder.pkl")
    artifact.add_file("scalar.pkl")
    run.log_artifact(artifact)
    run.finish()


def load_encoders_from_wanddb(project='IFT6758-2025-B01', artifact_name='preprocessing:latest') -> None:
    """
    Downloads the encoder + scalar artifact from W&B and returns them. We can then apply them into our test sets
    """
    run = wandb.init(project=project)
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()
    one_hot_encoder = joblib.load(f"{artifact_dir}/one_hot_encoder.pkl")
    scalar = joblib.load(f"{artifact_dir}/scalar.pkl")
    run.finish()
    return one_hot_encoder, scalar