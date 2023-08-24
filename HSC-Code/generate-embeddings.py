#!/usr/bin/env python3

import transformers
import gensim
from tqdm import tqdm
import numpy


def generate_w2v_model() -> gensim.models.KeyedVectors:
    tokenizer = transformers.AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    model = transformers.AutoModelForMaskedLM.from_pretrained("UBC-NLP/MARBERT")

    word_indices = tokenizer.vocab
    indices_word = list(
        map(lambda x: x[0], sorted(tokenizer.vocab.items(), key=lambda y: y[1]))
    )

    embeddings = list(model.modules())[2]
    word_embeddings = list(embeddings.modules())[1].weight.detach().numpy()

    model = gensim.models.KeyedVectors(
        word_embeddings.shape[1], count=word_embeddings.shape[0]
    )
    model.key_to_index = word_indices
    model.index_to_key = indices_word
    model.vectors = word_embeddings
    return model


def generate_top_k(wv: gensim.models.KeyedVectors, k: int = 10):
    top_k = numpy.zeros((wv.vectors.shape[0], k))
    index = gensim.similarities.MatrixSimilarity(
        gensim.matutils.Dense2Corpus(wv.vectors.T)
    )
    for i, word in tqdm(enumerate(index)):
        top_k[i, :] = word.argpartition(-k)[-k:]
    return top_k


W2V_MODEL_NAME = "marbert_w2v.model"
TOP_K_NAME = "marbert_top10"


def main():
    model = generate_w2v_model()
    model.save(W2V_MODEL_NAME)
    print("Generated w2v model. Generating top-k model")
    top_k = generate_top_k(model)
    numpy.save(TOP_K_NAME, top_k)
    print("Generated top-k examples.")


if __name__ == "__main__":
    main()
