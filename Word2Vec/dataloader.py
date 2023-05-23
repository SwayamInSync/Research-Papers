import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import spacy
from tqdm import tqdm
import random

random.seed(42)

spacy_en = spacy.load('en_core_web_sm')


class Vocabulary:
    def __init__(self, frequency_threshold):
        self.itos = {}
        self.stoi = {}

        self.frequency_threshold = frequency_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return []

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 0
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.frequency_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def encode(self, text, target=False):
        tokenized_text = self.tokenizer(text)
        encoded_text = [self.stoi[token] for token in tokenized_text]

        encoded_text = torch.tensor(encoded_text)
        if target:
            one_hot_encoding = torch.nn.functional.one_hot(
                encoded_text, num_classes=len(self.stoi))
            return one_hot_encoding

        return encoded_text

    def un_encode(self, encoding):
        labels = torch.argmax(encoding, dim=1)
        return " ".join([
            self.itos[token.data.item()] if token.data.item(
            ) in self.itos else self.itos[3]
            for token in labels
        ])


class EngVocabulary(Vocabulary):
    def __init__(self, frequency_threshold):
        super().__init__(frequency_threshold)

    @staticmethod
    def tokenizer(text):
        return [token.text.lower() for token in spacy_en.tokenizer(text) if
                not token.is_punct
                and not token.is_currency
                and not token.is_digit
                and not token.is_space]


class CustomDataset(Dataset):
    def __init__(self, root_dir, frequency_threshold_en=1, vocab=None, window_size=3):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        self.w = window_size
        self.english = open(os.path.join(
            root_dir, "data1.txt")).read().split("\n")[:-1]

        if vocab is None:
            self.vocab_en = EngVocabulary(frequency_threshold_en)
            self.vocab_en.build_vocabulary(self.english)
        else:
            self.vocab_en = vocab[0]

    def __len__(self):
        return len(self.english)

    def get_pairs(self, encoding):
        """
        params
        - encoding: (T,V) size array containing one_hot vectors of word
        :return: tensor of (num_words, T, 2*W+1, V) size tensor representing
        word, and it's context words having word itself at 0th index
        """
        num_words, vocab_len = encoding.shape
        pairs = torch.ones((num_words, 2 * self.w + 1, vocab_len)) * -1
        for i in range(num_words):
            pairs[i, 0] = encoding[i]

            if i < self.w:
                k = 1
                # filling left part
                words_in_left = i % self.w
                while (k <= words_in_left):
                    pairs[i, k] = encoding[i - k]
                    k += 1

                # filling right part
                if num_words - i - 1 >= self.w:
                    while (k <= i + self.w):
                        pairs[i, k] = encoding[k]
                        k += 1
                else:
                    while (k < num_words):
                        pairs[i, k] = encoding[k]
                        k += 1
            else:
                k = 1
                # filling left part
                while (k <= self.w):
                    pairs[i, k] = encoding[i - k]
                    k += 1
                # filling right part
                j = k
                k = i + 1
                if num_words - i - 1 >= self.w:
                    while (k <= i + self.w):
                        # print(i, k, num_words, self.w, torch.argmax(encoding[k]))
                        pairs[i, j] = encoding[k]
                        k += 1
                        j += 1
                else:
                    while (k < num_words):
                        # print(i, k, num_words, self.w, torch.argmax(encoding[k]))
                        pairs[i, j] = encoding[k]
                        k += 1
                        j += 1

        # shuffling the pairs
        # pairs = pairs[torch.randperm(pairs.shape[0])]
        return pairs

    def __getitem__(self, index):
        english_sentence = self.english[index]
        one_hot_encoded = self.vocab_en.encode(english_sentence)
        pairs = self.get_pairs(one_hot_encoded)
        return pairs


def get_loader(root_dir, batch_size, shuffle, vocab=None, window_size=5):
    dataset = CustomDataset(root_dir, vocab=vocab, window_size=window_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle,
    )
    return dataset, loader


def make_words():
    train_set, train_loader = get_loader(
        os.getcwd(), batch_size=1, shuffle=False, window_size=5)
    vocab = train_set.vocab_en
    for i in tqdm(train_loader, total=len(train_loader), leave=False):
        words = []
        i = i.squeeze()
        center_word = i[:, 0, :]
        context = i[:, 1:, :]
        center_word = torch.argmax(center_word, dim=1)
        context = torch.argmax(context, dim=2)
        for a in range(context.shape[0]):
            c_word = center_word[a].item()
            for b in range(context.shape[1]):
                if i[a, b, context[a, b]] != -1:
                    words.append(
                        [vocab.itos[c_word], vocab.itos[context[a, b].item()]])
            words = words[:-1]
        df = pd.DataFrame(words)
        df.to_csv('data.csv', index=False, mode='a',
                  columns=None, header=False)
        del df, words
    return train_set, train_loader


class WordDataset(Dataset):
    def __init__(self, vocab):
        super(WordDataset, self).__init__()
        self.vocab = vocab
        self.pairs = pd.read_csv("data.csv", header=None)

    def get_data(self):
        return self.pairs

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, index):
        return self.vocab.encode(self.pairs.iloc[index][0]).squeeze(), self.vocab.encode(self.pairs.iloc[index][1], target=True).squeeze().double()


def get_word_loader(batch_size, shuffle):
    path = os.path.join(os.getcwd(), 'data.csv')
    if not os.path.exists(path):
        train_set, _ = make_words()
    else:
        train_set, _ = get_loader(
            os.getcwd(), batch_size=1, shuffle=False, window_size=5)
    vocab = train_set.vocab_en
    data = WordDataset(vocab)
    wordloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return data, wordloader


if __name__ == "__main__":
    wordset, wordloader = get_word_loader(32, True)
    for i in tqdm(wordloader, total=len(wordloader), leave=False):
        try:
            i[0].shape
            i[1].shape
        except KeyError:
            print(i[0].shape, i[1].shape)
