import os
from sklearn.model_selection import train_test_split
import numpy as np
from operator import itemgetter
from math import log
import random
from gensim.summarization.summarizer import summarize
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KDTree
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import pandas as pd

def add_to_dict(D, k, v=1):
    if k in D:
        D[k] += v
    else:
        D[k] = v

def clean_string(string):
    out_string = ""
    for c in string:
        if c.isalpha():
            out_string += c
        else:
            if len(out_string) > 0 and out_string[-1] != " ":
                out_string += " "
    return out_string

def read_topic_list(topic_file):
    topic_words = {}
    for row in topic_file:
        fields = row[:-1].split("|")
        topic_id = int(fields[0])
        words = fields[1].split(",")
        topic_words[topic_id] = words
    topic_file.close()
    return topic_words

