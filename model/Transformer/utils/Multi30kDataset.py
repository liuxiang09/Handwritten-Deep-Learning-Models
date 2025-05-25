from torch.utils.data import Dataset, DataLoader
import torch

def collate_fn(batch, src_pad_idx, trg_pad_idx):
    # batch 是一个列表，每个元素是 __getitem__ 返回的 (src_tensor, trg_tensor)
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(src_item)
        trg_batch.append(trg_item)

    # 使用 torch.nn.utils.rnn.pad_sequence 进行填充
    # pad_sequence 函数接收一个Tensor或者一个List[Tensor]，填充pad后输出一个Tensor，形状由batch_first参数决定
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch,
                                                  batch_first=True,
                                                  padding_value=src_pad_idx)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_batch,
                                                  batch_first=True,
                                                  padding_value=trg_pad_idx)

    return src_padded, trg_padded

class Multi30kDataset(Dataset):
    def __init__(self, 
                 src_filepath, 
                 trg_filepath, 
                 src_tokenizer, 
                 trg_tokenizer, 
                 max_seq_len,
        ):
        
        self.src_sentences = self._read_sentences(src_filepath)
        self.trg_sentences = self._read_sentences(trg_filepath)
        # 源文件句子数 == 目标文件句子数
        assert len(self.src_sentences) == len(self.trg_sentences), \
            "Source and target files must have the same number of lines."

        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_seq_len = max_seq_len

        # 特殊的token id
        self.src_sos_idx = self.src_tokenizer.bos_token_id
        self.src_eos_idx = self.src_tokenizer.eos_token_id
        self.src_pad_idx = self.src_tokenizer.pad_token_id
        self.trg_sos_idx = self.trg_tokenizer.bos_token_id
        self.trg_eos_idx = self.trg_tokenizer.eos_token_id
        self.trg_pad_idx = self.trg_tokenizer.pad_token_id

    def _read_sentences(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        trg_sentence = self.trg_sentences[idx]

        # add_special_tokens=False: 我们手动添加EOS，避免tokenizer自动添加不需要的CLS/SEP
        src_numerical = self.src_tokenizer.encode(
            src_sentence,
            add_special_tokens=False,  # 不让HF tokenizer自动加 CLS 和 SEP
            truncation=True, max_length=self.max_seq_len - 1  # 预留<eos>的地方
        )
        trg_numerical = self.trg_tokenizer.encode(
            trg_sentence,
            add_special_tokens=False,
            truncation=True, max_length=self.max_seq_len - 2 # 预留<sos>, <eos>的地方
        )

        src_numerical = src_numerical + [self.src_eos_idx]
        trg_numerical = [self.trg_sos_idx] + trg_numerical + [self.trg_eos_idx]

        return torch.tensor(src_numerical, dtype=torch.long), \
               torch.tensor(trg_numerical, dtype=torch.long)
        

