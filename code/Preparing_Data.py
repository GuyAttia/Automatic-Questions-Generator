from collections import Counter
import csv
import numpy as np

from embedding import look_up_word, START_WORD, END_WORD

_MAX_BATCH_SIZE = 64


def _tokenize(string):
    return [word.lower() for word in string.split(" ")]


def _prepare_batch(batch):
    id_to_indices = {}
    document_ids = []
    document_text = []
    document_words = []
    answer_text = []
    answer_indices = []
    question_text = []
    question_input_words = []
    question_output_words = []
    for i, entry in enumerate(batch):
        id_to_indices.setdefault(entry["document_id"], []).append(i)
        document_ids.append(entry["document_id"])
        document_text.append(entry["document_text"])
        document_words.append(entry["document_words"])
        answer_text.append(entry["answer_text"])
        answer_indices.append(entry["answer_indices"])
        question_text.append(entry["question_text"])
        question_words = entry["question_words"]
        question_input_words.append([START_WORD] + question_words)
        question_output_words.append(question_words + [END_WORD])

    batch_size = len(batch)
    max_document_len = max(len(document) for document in document_words)
    max_answer_len = max(len(answer) for answer in answer_indices)
    max_question_len = max(len(question) for question in question_input_words)

    document_tokens = np.zeros((batch_size, max_document_len), dtype=np.int32)
    document_lengths = np.zeros(batch_size, dtype=np.int32)
    answer_labels = np.zeros((batch_size, max_document_len), dtype=np.int32)
    answer_masks = np.zeros((batch_size, max_answer_len, max_document_len), dtype=np.int32)
    answer_lengths = np.zeros(batch_size, dtype=np.int32)
    question_input_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_output_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_lengths = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        for j, word in enumerate(document_words[i]):
            document_tokens[i, j] = look_up_word(word)
        document_lengths[i] = len(document_words[i])

        for j, index in enumerate(answer_indices[i]):
            for shared_i in id_to_indices[batch[i]["document_id"]]:
                try:
                    answer_labels[shared_i, index] = 1
                except:
                    answer_labels[shared_i, index-1] = 1
            try:
                answer_masks[i, j, index] = 1
            except:
                answer_masks[i, j, index-1] = 1
        answer_lengths[i] = answer_indices[i][-1] - answer_indices[i][0] + 1

        for j, word in enumerate(question_input_words[i]):
            question_input_tokens[i, j] = look_up_word(word)
        for j, word in enumerate(question_output_words[i]):
            question_output_tokens[i, j] = look_up_word(word)
        question_lengths[i] = len(question_input_words[i])

    return {
        "size": batch_size,
        "document_ids": document_ids,
        "document_text": document_text,
        "document_words": document_words,
        "document_tokens": document_tokens,
        "document_lengths": document_lengths,
        "answer_text": answer_text,
        "answer_indices": answer_indices,
        "answer_labels": answer_labels,
        "answer_masks": answer_masks,
        "answer_lengths": answer_lengths,
        "question_text": question_text,
        "question_input_tokens": question_input_tokens,
        "question_output_tokens": question_output_tokens,
        "question_lengths": question_lengths,
    }


def _read_data(path):
    stories = {}

    with open(path, encoding='utf-8', errors='replace') as f:
        header_seen = False
        for row in csv.reader(f):
            if not header_seen:
                header_seen = True
                continue
    
            document_id = row[0]
    
            existing_stories = stories.setdefault(document_id, [])
    
            document_text = row[1]
            if existing_stories and document_text == existing_stories[0]["document_text"]:
                # Save memory by sharing identical documents
                document_text = existing_stories[0]["document_text"]
                document_words = existing_stories[0]["document_words"]
            else:
                document_words = _tokenize(document_text)
    
            question_text = row[2]
            question_words = _tokenize(question_text)
            
            answer_text = row[3]
            answer_words = _tokenize(answer_text)
            answer_start_ind = int(row[4])
            earlier_words_ind = len(_tokenize(document_text[:answer_start_ind])) -1
            answer_end_ind = earlier_words_ind + len(answer_words)
            answer_indices = list(range(earlier_words_ind, answer_end_ind))

            existing_stories.append({
                "document_id": document_id,
                "document_text": document_text,
                "document_words": document_words,
                "answer_text": answer_text,
                "answer_indices": answer_indices,
                "question_text": question_text,
                "question_words": question_words,
            })
      
    return stories


def _process_stories(stories):
    batch = []
    for story in stories.values():
        if len(batch) + len(story) > _MAX_BATCH_SIZE:
            yield _prepare_batch(batch)
            batch = []
        batch.extend(story)
    if len(batch) == _MAX_BATCH_SIZE:
        yield _prepare_batch(batch)


_training_stories = None
_test_stories = None
train_path = 'Data/Train_DataSet-clean.csv'
test_path = 'Data/Test_DataSet-clean.csv'


def _load_training_stories():
    global _training_stories
    if not _training_stories:
        _training_stories = _read_data(train_path)
    return _training_stories


def _load_test_stories():
    global _test_stories
    if not _test_stories:
        _test_stories = _read_data(test_path)
    return _test_stories


def training_data():
    return _process_stories(_load_training_stories())


def test_data():
    return _process_stories(_load_test_stories())


def trim_embeddings():
    document_counts = Counter()
    question_counts = Counter()
    for data in [_load_training_stories().values(), _load_test_stories().values()]:
        for stories in data:
            document_counts.update(stories[0]["document_words"])
            for story in stories:
                question_counts.update(story["question_words"])

    keep = set()
    for word, count in question_counts.items():
        keep.add(word)
    for word, count in document_counts.items():
        keep.add(word)
 
    # Create a trimmed file using only the 20000 words we keeped
    with open('GloVe/glove.6B.300d.txt',encoding="utf-8") as f:
        with open('GloVe/glove.6B.300d.v2.txt', "w",encoding="utf-8") as f2:
            for line in f:
                if line.split(" ")[0] in keep:
                    f2.write(line)


if __name__ == '__main__':
    trim_embeddings()
