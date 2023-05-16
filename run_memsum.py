from memsum import MemSum
from nltk import sent_tokenize

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str)

args = parser.parse_args()

model_path = "model/MemSum_Final/model.pt"

opinion = " ".join(open(args.filename, "r").readlines())

def preprocess(text):
    text = text.replace("\n","")
    text = text.replace("¶","")
    text = " ".join(text.split())
    return text


opinion = sent_tokenize( preprocess(opinion) )

summarizer = MemSum(model_path, "model/glove/vocabulary_200dim.pkl").summarize

summary = "\n".join( summarizer(opinion) )

print(summary)