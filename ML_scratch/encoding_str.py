import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. split words in entire
        entire = positive + negative
        words = set()
        for sent in entire:
            for word in sent.split():
                words.add(word)
        
        # 2. sorting in alphabetic -> mapping
        word_to_int = {} # words: int
        for i, word in enumerate(sorted(list(words))):
            word_to_int[word] = i+1

        # 3. encoding
        def encode(sent):
            integers = []
            for word in sent.split():
                integers.append(word_to_int[word])
            return integers
        final = []
        for sent in entire:
            final.append(torch.tensor(encode(sent)))

        # 4. padding to the maximum length
        return nn.utils.rnn.pad_sequence(final, batch_first=True)


        
        
