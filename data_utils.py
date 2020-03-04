import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class WordData(Dataset):

    def __init__(self, tags_data, lems_data, wf_data,
                 tag2index, char2index,
                 sequence_length=32,
                 tag_pad_token='TPAD', lem_pad_token='CPAD', wf_pad_token='CPAD',
                 verbose=True):

        super().__init__()

        self.tags_data = []
        self.lems_data = []
        self.wf_data = []

        self.tag2index = tag2index
        self.char2index = char2index

        self.sequence_length = sequence_length
        # self.tags_sequence_length = sequence_length
        # self.lems_sequence_length = sequence_length  # todo: check
        # self.wf_sequence_length = sequence_length  # todo: check

        self.tags_pad_token = tag_pad_token
        self.lems_pad_token = lem_pad_token
        self.wf_pad_token = wf_pad_token  # todo: should it be different from lem one

        self.tags_pad_index = self.tag2index[self.tags_pad_token]
        self.lems_pad_index = self.char2index[self.lems_pad_token]
        self.wf_pad_index = self.char2index[self.wf_pad_token]

        data = zip(tags_data, lems_data, wf_data)
        self.load(data, verbose=verbose)

    @staticmethod
    def process_tags(tags):
        tags_li = tags  # tags.split(',')
        return tags_li

    @staticmethod
    def process_lem(lem):
        chars = list(lem)
        return chars

    @staticmethod
    def process_wf(wf):
        chars = list(wf)
        return chars

    def load(self, data, verbose=True):

        data_iterator = tqdm(data, desc='Loading data', disable=not verbose)

        for tags, lem, wf in data_iterator:
            tags = self.process_tags(tags)
            lem = self.process_lem(lem)
            wf = self.process_wf(wf)

            indexed_tags = self.indexing_tags(tags)
            indexed_lem = self.indexing_lem(lem)
            indexed_wf = self.indexing_wf(wf)

            self.tags_data.append(indexed_tags)
            self.lems_data.append(indexed_lem)
            self.wf_data.append(indexed_wf)

    def indexing_tags(self, tags):
        return [self.tag2index[tag] for tag in tags if tag in self.tag2index]

    def indexing_lem(self, lem):
        return [self.char2index[c] for c in lem if c in self.char2index]

    def indexing_wf(self, wf):
        return [self.char2index[c] for c in wf if c in self.char2index]

    def padding(self, sequence, padding_symbol_index):

        # Ограничить длину self.sequence_length
        # если длина меньше максимально - западить

        ### CODE ###

        if len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        elif len(sequence) < self.sequence_length:
            sequence = sequence + [padding_symbol_index] * (self.sequence_length - len(sequence))

        return sequence

    def __len__(self):

        return len(self.tags_data)

    def __getitem__(self, idx):

        tags = self.tags_data[idx]
        tags = self.padding(tags, self.tags_pad_index)
        tags = torch.Tensor(tags).long()

        lem = self.lems_data[idx]
        lem = self.padding(lem, self.lems_pad_index)
        lem = torch.Tensor(lem).long()

        wf = self.wf_data[idx]
        wf = self.padding(wf, self.wf_pad_index)
        wf = torch.Tensor(wf).long()

        x = (tags, lem)
        y = wf
        return x, y