from memsum import MemSum
from nltk import sent_tokenize

model_path = "model/MemSum_Final/model.pt"

opinion = " ".join(open("opinion.txt", "r").readlines())

def preprocess(text):
    text = text.replace("\n","")
    text = text.replace("¶","")
    text = " ".join(text.split())
    return text


opinion = sent_tokenize( preprocess(opinion) )

summarizer = MemSum(model_path, "model/glove/vocabulary_200dim.pkl").summarize

summary = "\n".join( summarizer(opinion) )

print(summary)