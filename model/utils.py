import torch
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index (Sentence aware) """

    def __init__(self, character):
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1  # 0 reserved for CTC blank

        self.character = ['[CTCblank]'] + dict_character

    def encode(self, text, batch_max_length=120):
        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)

        for i, t in enumerate(text):
            t = list(t)
            t = [self.dict[char] for char in t]
            batch_text[i][:len(t)] = torch.LongTensor(t)

        return batch_text.to(device), torch.IntTensor(length).to(device)

    def decode(self, text_index, length):
        texts = []

        for index, l in enumerate(length):
            t = text_index[index, :]
            char_list = []
            prev_idx = None

            for i in range(l):
                cur_idx = t[i].item()
                if cur_idx != 0 and cur_idx != prev_idx:
                    char_list.append(self.character[cur_idx])
                prev_idx = cur_idx

            text = ''.join(char_list)
            text = re.sub(r'\s+', ' ', text).strip()
            texts.append(text)

        return texts


def simple_language_postprocess(text):
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
    return text


class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        return self.sum / float(self.n_count) if self.n_count != 0 else 0
