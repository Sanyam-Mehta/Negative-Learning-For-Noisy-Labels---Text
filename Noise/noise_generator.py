import pandas as pd
import numpy as np
import random

TEXT_COL, LABEL_COL = 'text', 'label'


def generate_noise_symminc(df, num_classes, noise_ratio):
    clean_df, noise_df = get_split(df, noise_ratio)
    noise_df[LABEL_COL] = noise_df[LABEL_COL].apply(lambda x: random.sample(range(num_classes), k=1)[0])
    return pd.concat((clean_df, noise_df), axis=0)


def generate_noise_symmexc(df, num_classes, noise_ratio):
    clean_df, noise_df = get_split(df, noise_ratio)
    noise_df[LABEL_COL] = noise_df[LABEL_COL].apply(lambda x: (x + random.sample(range(1, num_classes), k=1)[0])%num_classes)
    return pd.concat((clean_df, noise_df), axis=0)


def get_split(df, noise_ratio):
    clean_df, noise_df = np.split(df.sample(frac=1), [int((1. - noise_ratio) * len(df))])
    return clean_df, noise_df


def generate_noise(df, noise_type="inc", num_classes=6, noise_ratio=0.3):
    if noise_type == "inc":
        return generate_noise_symminc(df, num_classes, noise_ratio)
    if noise_type == "exc":
        return generate_noise_symmexc(df, num_classes, noise_ratio)


