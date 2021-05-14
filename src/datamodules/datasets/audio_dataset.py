import os
import pathlib

import pandas as pd
import torchaudio
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing import List

from src.utils.audio_utils import resample


class AudioDataset(Dataset):
    """Dataclass to load wavfiles from folder."""

    def __init__(self, dir: str, sr: int, classes: List):
        self.data_path = pathlib.Path(dir)
        assert self.data_path.exists()
        self.sr = sr
        self.classes = classes
        self.df = self.get_dataset_dataframe()

    def __getitem__(self, idx: int):
        row = self.df.loc[idx, :]
        audiofile_path = self.data_path.joinpath(row["relative_path"])
        class_ids = row["class_ids"]

        sig, sr = torchaudio.load(audiofile_path)
        if sr is not self.sr:
            sig, sr = resample((sig, sr), self.sr)

        return sig, class_ids

    def __len__(self):
        return len(self.df)

    def get_dataset_dataframe(self):
        """Creates Dataframe for Dataset. Scans self.data_path for wav files and corresponding csv.

        Returns:
            pd.DataFrame: Dataframe with columns "relative_path" and "class_ids"
        """
        wav_paths = [_ for _ in self.data_path.iterdir() if _.suffix == ".wav"]
        label_paths = [_.with_suffix(".txt") for _ in wav_paths]
        labels = [self.read_label_txt(_)["label"].to_list() for _ in tqdm(label_paths)]
        mlb = MultiLabelBinarizer(classes=self.classes)
        label_array = mlb.fit_transform(labels)
        classes = mlb.classes_

        df = pd.DataFrame(
            data={
                "relative_path": [_.name for _ in wav_paths],
                "class_ids": [_ for _ in label_array],
            }
        )

        return df

    def read_label_txt(self, path: pathlib.Path) -> pd.DataFrame:
        colnames = ["onset", "offset", "label"]
        df = pd.read_csv(
            path, sep="\t", names=["onset", "offset", "label"], index_col=False
        )
        return df



