import numpy as np

# embeddings_path = 'GloVe/glove.6B.300d.v2.txt'
embeddings_path = 'GloVe/glove.6B.100d.trimmed.txt'
_word_to_idx = {}
_idx_to_word = []


def _add_word(word):
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx


UNKNOWN_WORD = "<UNK>"
START_WORD = "<START>"
END_WORD = "<END>"

UNKNOWN_TOKEN = _add_word(UNKNOWN_WORD)
START_TOKEN = _add_word(START_WORD)
END_TOKEN = _add_word(END_WORD)


def look_up_word(word):
    return _word_to_idx.get(word, UNKNOWN_TOKEN)


def look_up_token(token):
    return _idx_to_word[token]


def create_glove():
    with open(embeddings_path, encoding="utf-8") as f:
    # Find the dimension and size(number of words) of the vocabulary(glove.6B.300d.v2 file)
        line = f.readline()
        chunks = line.split(" ")
        dimensions = len(chunks) - 1
        f.seek(0)
        vocab_size = sum(1 for line in f)
        vocab_size += 3
        f.seek(0)

        # Build vectors for all the unknown words
        unknown_words_path = 'Data/Unknown_Words.txt'
        unknown_words = {}
        with open(unknown_words_path, encoding="utf-8") as u:
            for line in u.readlines():
                word = line[:-1]
                unknown_words[word] = np.random.rand(dimensions)

        vocab_size += len(unknown_words)

    # Build the glove(embedding) numpy array
        glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
        glove[UNKNOWN_TOKEN] = np.zeros(dimensions) # 0 for unknown token
        glove[START_TOKEN] = -np.ones(dimensions)   # -1 for the start token
        glove[END_TOKEN] = np.ones(dimensions)      # 1 for the end token

        for line in f:
            chunks = line.split(" ")
            idx = _add_word(chunks[0])
            glove[idx] = [float(chunk) for chunk in chunks[1:]]

        for word,arr in unknown_words.items():
            idx = _add_word(word)
            glove[idx] = arr

        # Save the glove array for later use
        # np.save('GloVe/glove', glove)

        return glove

glove = create_glove()

if __name__ == '__main__':
    create_glove()
