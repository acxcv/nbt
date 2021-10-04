from torch.utils.data import Dataset
import pandas as pd
from word_vector import load_obj
from torch.utils.data import DataLoader


class OurDataset(Dataset):
    def __init__(self, pickle_filename):
        self.data = load_obj(pickle_filename)  # list with all the tensors

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        return {'sys_out': self.data.iloc[idx]['previous_system_output'],
                'user_utt': self.data.iloc[idx]['vectorized_utterance'],
                'cand_pair': self.data.iloc[idx]['candidate_pair'],
                'binary_label': self.data.iloc[idx]['binary_label']}


def create_data_loader(pickle_path, n_batches=256):
    data = OurDataset(pickle_path)
    dataloader = DataLoader(data, batch_size=n_batches, shuffle=True,
                            collate_fn=custom_collate)

    return dataloader


def custom_collate(x):
    return x
