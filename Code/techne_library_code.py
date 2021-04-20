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

def read_topic_list(file_name):
    topic_words = {}
    topic_file = open(file_name, 'r')
    for row in topic_file:
        fields = row[:-1].split("|")
        topic_id = int(fields[0])
        words = fields[1].split(",")
        topic_words[topic_id] = words
    topic_file.close()
    return topic_words

def read_doc_topics(file_name):
    topics_per_doc = {}
    doc_topics = open(file_name, 'r')
    for row in doc_topics:
        fields = row[:-1].split("|")
        file_name = fields[0]
        topic_probs = [float(x) for x in fields[1:]]
        topics_per_doc[file_name] = topic_probs
    doc_topic_file.close()
    return topics_per_doc

def plot_doc_topics(doc_ids, doc_topic_lookup, topic_count):
    fig, ax = pyplot.subplots(2,2)
    fig.set_size_inches(8.5,5)
    for i, file_number in enumerate(doc_ids):
        topic_probs = doc_topic_lookup["file_" + str(file_number) + ".txt"]
        ax[int(i/2), i % 2].bar(x = [str(x) for x in range(topic_count)], height = topic_probs)
    return fig, ax

