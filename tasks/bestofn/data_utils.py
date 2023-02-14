from torch.utils.data import Dataset


class BestOfNDataset(Dataset):
    def __init__(self, name: str, data_prefix: str, max_seq_length: int):
        raise NotImplementedError("读取 preprocess_data_best_of_n.py 生成的数据集")

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError("""collate preprocess_data_best_of_n.py 生成的数据集
        
数据集的结果返回为
  * samples:  [batch_size, N, seq_len]
  * label: [batch_size]
  * mask: [batch_size, N, seq_len]
  
其中， prompt拼到 samples中。padding value可以是任何值，反正会被mask掉。
""")
