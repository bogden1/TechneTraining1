import pandas as pd
import spacy
from functools import reduce


def join_strings(x, y):
    return str(x) + " " + str(y)


def extract_orgs(doc):
    for entity in doc.ents:
        if entity.label_ == "ORG":
            orgs.append(entity.text)


def process(chunk):
    for df in chunk:
        text = reduce(join_strings, df[1])
        doc = nlp(text)
        extract_orgs(doc)


def read_chunk(fname, chunk=100, rows=100000):
    df = pd.read_csv(
        fname,
        header=None,
        skiprows=0,
        chunksize=chunk,
        nrows=rows,
        compression="gzip",
        sep="|",
    )
    return df


orgs = []

nlp = spacy.load("en_core_web_sm")

chunk = read_chunk("../../Data/regulation_search_results_10000.txt.gz", rows=10000)
process(chunk)
