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
import ipywidgets as widgets
from matplotlib.colors import LogNorm


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
    doc_topics.close()
    return topics_per_doc

def normalise_vector(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0: 
       return v
    return v / norm

def plot_doc_topics(doc_ids, doc_topic_lookup, topic_count, normalise=True):
    fig, ax = pyplot.subplots(2,2)
    fig.set_size_inches(8.5,5)
    for i, file_number in enumerate(doc_ids):
        topic_probs = doc_topic_lookup["file_" + str(file_number) + ".txt"]
        if normalise:
            topic_probs = normalise_vector(topic_probs)
        ax[int(i/2), i % 2].bar(x = [str(x) for x in range(topic_count)], height = topic_probs)
    return fig, ax

def filter_topics_by_threshold(topic_dict, threshold):
    filtered_dict = {}
    for k,v in topic_dict.items():
        scores = [x if x >= threshold else 0.0 for i,x in enumerate(v)]
        filtered_dict[k] = scores
    return filtered_dict

def topic_to_class_scores(topic_scores, topic_class_map):
    file_class_scores = {}
    max_class = max([v for v in topic_class_map.values()])
    for doc_id,scores in topic_scores.items():
        class_scores = np.zeros(max_class+1)
        for t,s in enumerate(scores):
            class_scores[topic_class_map[t]] += s
        file_class_scores[doc_id] = class_scores
    return file_class_scores

def load_content_file_map(file_name):
    file_domain = {}
    file_map = open(data_drive + "TM/content_file_map.txt","r")
    file_url = {}
    for row in file_map:
        fields = row[:-1].split("|")
        file_url[fields[0]] = fields[1]
        file_domain[fields[0]] = fields[1].split("/")[0]
    file_map.close()

def load_content(file_name):
    content_file = open(file_name, "r")
    file_contents = {}
    for row in content_file:
        fields = row[:-1].split("|")
        file_contents[fields[0]] = fields[1]
    content_file.close()
    return file_contents

def load_summaries(file_name):
    summary_file = open(file_name, 'r')
    file_summaries = {}
    for row in summary_file:
        fields = row[:-1].split("|")
        file_summaries[fields[0]] = fields[1]
    summary_file.close()
    return file_summaries

class MLData:

    def __init__(self):
        self.corpus = []
        self.file_contents = {}
        self.file_to_idx = {}
        self.stop_words = []
        self.file_classes = {}

    def clean_string(self,string):
        out_string = ""
        for c in string:
            if c.isalpha():
                out_string += c
            else:
                if len(out_string) > 0 and out_string[-1] != " ":
                    out_string += " "
        return out_string

    def add_stop_words(self, *stop_words):
        for w in stop_words:
            self.stop_words.append(w)

    def load_content(self,file_name):
        content_file = open(file_name, "r")
        self.file_contents = {}
        self.corpus = []
        self.file_to_idx = {}
        for row in content_file:
            fields = row[:-1].split("|")
            if len(fields[1]) == 0:
                continue
            self.corpus.append(self.clean_string(fields[1].lower()))
            self.file_to_idx[fields[0]] = len(self.corpus)
            self.file_contents[fields[0]] = fields[1]
            self.file_classes[fields[0]] = -1
        content_file.close()

    def set_classes(self, file_classes):
        for k,v in file_classes.items():
            self.file_classes[k] = v

    def get_tfidf(self, features, min_df, max_df):
        self.vectorizer = TfidfVectorizer(max_features=features, min_df=min_df, max_df=max_df, stop_words = self.stop_words)
        self.TFIDF = vectorizer.fit_transform(corpus)


def prepare_for_ml(tfidf_features, classes_per_doc, file_to_idx_map):
    training_files = []
    training_features = []
    training_class = []
    feature_matrix = tfidf_features.todense()

    for filename, scores in classes_per_doc.items():
        norm_scores = normalise_vector(scores)
        highest = np.argmax(norm_scores)
        training_files.append(filename)
        training_class.append(highest)
        training_features.append(feature_matrix[file_to_idx_map[filename]])
    training_features = np.vstack(training_features)

    return training_files, training_features, training_class

def draw_confusion(y_true, y_pred, model, class_names):
    fig, ax = pyplot.subplots(1,1,figsize=(7, 7))
    N = len(model.classes_)
    sns.heatmap(pd.DataFrame(confusion_matrix(y_true, y_pred, normalize=None),
                             range(N), range(N)), cmap='magma', annot=True, annot_kws={"size": 15}, fmt='g', ax = ax) #, norm=LogNorm())
    #ax.table(cellText=topN[{'TaxonomyCategory','TAXID'}].sort_values(by='TAXID').values, colLabels=['TaxonomyCategory','TAXID'], loc='top')
    ax.set_xticklabels([class_names[c] for c in model.classes_])
    ax.set_yticklabels([class_names[c] for c in model.classes_], rotation = 30)
    return fig, ax
