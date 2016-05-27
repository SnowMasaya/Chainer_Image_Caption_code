#!/usr/bin/env python
import os
import numpy as np
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) ) + "/../"


class Read_Data():

    def __init__(self, image_id_file, root, caption_file):
        self.image_id_file = image_id_file
        self.root = root
        self.caption_file = caption_file
        self.vocab = {}
        self.total_words_ids = {}
        self.images_ids = {}
        self.id_image_tuples = []
        self.dataset = np.ndarray([], dtype=np.int32)

    def load_image_list(self):
        image_id_file = open(self.image_id_file)
        for line in image_id_file:
            imgID, pair = line.strip().split(",")

            if imgID not in self.images_ids:
                self.images_ids[imgID] = os.path.join(self.root, pair)
            self.id_image_tuples.append((os.path.join(self.root, pair),
                                         np.int32(imgID)))
        image_id_file.close()

    def load_caption_list(self):
        total_words = []
        caption_file = open(self.caption_file)
        for lines in caption_file:
            if len(lines) == 1: continue 
            img_caption_ID, words = lines.strip().split("\t")
            if img_caption_ID not in self.total_words_ids:
                self.total_words_ids[img_caption_ID] = words
            words = words.split(" ")
            total_words.extend(words)
            total_words.extend('<eos>')

        self.dataset = np.ndarray((len(total_words),), dtype=np.int32)
        for i, word in enumerate(total_words):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            self.dataset[i] = self.vocab[word]
        caption_file.close()
