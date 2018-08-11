import random
import numpy as np
import csv
import nltk
from nltk.corpus import stopwords

from Preparing_Data import _tokenize, _MAX_BATCH_SIZE
from embedding import look_up_word

new_quest_path = 'Data/inference_questions.csv'
_new_stories = None


def NER(document, logger):
    """ This function extract answers and synonyms from document and save it to csv file in the new_quest_path """
    logger.info('Start extraction of sentences')
    sentences = parse_document(document, logger)

    logger.info('Start extraction of answers')
    chunked_sentences = nltk.ne_chunk_sents(sentences, binary=True)
    answers = set()
    for tree in chunked_sentences:
        answers.update(extract_entity_names(tree))
    logger.info('# of extracted answers = {}'.format(len(answers)))
    logger.info('Start generation of synonyms')
    synonyms = generate_synonyms(answers, sentences, logger)
    save_csv(document, answers, synonyms)


def parse_document(document, logger):
    document = ' '.join([i for i in document.split()])
    sentences = nltk.sent_tokenize(document)
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    tagged_sentences = [nltk.pos_tag(sent) for sent in tokenized_sentences]
    logger.info('# of extracted sentences = {}'.format(len(tagged_sentences)))
    return tagged_sentences


def extract_entity_names(tree):
    entity_names = set()
    if hasattr(tree, 'label') and tree.label:
        if tree.label() == 'NE':
            entity_names.add(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                entity_names.update(extract_entity_names(child))

    return entity_names


def generate_synonyms(answers, sentences, logger):
    stop = set(stopwords.words('english'))
    optional_words = []
    for sent in sentences:
        for word_pair in sent:
            if (word_pair[0].lower() not in stop) and (len(word_pair[0]) > 2):
                optional_words.append(word_pair[0])

    synonyms = []
    for answer in answers:
        optWords = optional_words
        if answer in optWords:
            optWords.remove(answer)
        randNums = random.sample(range(len(optWords)), 3)
        synonyms.append([optWords[i] for i in randNums])
    return synonyms


def save_csv(document, answers, synonyms):
    with open(new_quest_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['', 'Paragraph', 'Question', 'Answer', 'Ans Starting Index', 'Synonyms'])
        for i, ans in enumerate(answers):
            ans_ind = document.index(ans)
            writer.writerow([i, document, '', ans, ans_ind, synonyms[i]])


def new_quest_data():
    return _process_stories(_load_new_quest_stories())


def _load_new_quest_stories():
    global _new_stories
    if not _new_stories:
        _new_stories = _read_data(new_quest_path)
    return _new_stories


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
            earlier_words_ind = len(_tokenize(document_text[:answer_start_ind])) - 1
            answer_end_ind = earlier_words_ind + len(answer_words)
            answer_indices = list(range(earlier_words_ind, answer_end_ind))

            synonyms = row[5]

            existing_stories.append({
                "document_id": document_id,
                "document_text": document_text,
                "document_words": document_words,
                "answer_text": answer_text,
                "answer_indices": answer_indices,
                "question_text": question_text,
                "question_words": question_words,
                "synonyms": synonyms,
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
    elif len(batch) < _MAX_BATCH_SIZE:
        # Fill it up to max batch size and then yield:
        batch.extend([batch[-1]]*(_MAX_BATCH_SIZE-len(batch)))
        yield _prepare_batch(batch)


def _prepare_batch(batch):
    id_to_indices = {}
    document_ids = []
    document_text = []
    document_words = []
    answer_text = []
    answer_indices = []
    synonyms = []
    for i, entry in enumerate(batch):
        id_to_indices.setdefault(entry["document_id"], []).append(i)
        document_ids.append(entry["document_id"])
        document_text.append(entry["document_text"])
        document_words.append(entry["document_words"])
        answer_text.append(entry["answer_text"])
        answer_indices.append(entry["answer_indices"])
        synonyms.append(entry["synonyms"])

    batch_size = len(batch)
    max_document_len = max(len(document) for document in document_words)
    max_answer_len = max(len(answer) for answer in answer_indices)

    document_tokens = np.zeros((batch_size, max_document_len), dtype=np.int32)
    document_lengths = np.zeros(batch_size, dtype=np.int32)
    answer_labels = np.zeros((batch_size, max_document_len), dtype=np.int32)
    answer_masks = np.zeros((batch_size, max_answer_len, max_document_len), dtype=np.int32)
    answer_lengths = np.zeros(batch_size, dtype=np.int32)
    for i in range(batch_size):
        for j, word in enumerate(document_words[i]):
            document_tokens[i, j] = look_up_word(word)
        document_lengths[i] = len(document_words[i])
        for j, index in enumerate(answer_indices[i]):
            for shared_i in id_to_indices[batch[i]["document_id"]]:
                answer_labels[shared_i, index] = 1
            answer_masks[i, j, index] = 1
        answer_lengths[i] = answer_indices[i][-1] - answer_indices[i][0] + 1

    return {
        "size": batch_size,
        "document_text": document_text,
        "document_tokens": document_tokens,
        "document_lengths": document_lengths,
        "answer_labels": answer_labels,
        "answer_masks": answer_masks,
        "answer_lengths": answer_lengths,
        "answer_text": answer_text,
        "synonyms": synonyms,
    }


if __name__ == '__main__':
    para_text = "Super Bowl 50 was an American football game to determine the " \
                "champion of the National Football League (NFL) for the 2015 season. " \
                "The American Football Conference (AFC) champion Denver Broncos defeated the " \
                "National Football Conference (NFC) champion Carolina Panthers 24–10 to " \
                "earn their third Super Bowl title. The game was played on February 7, 2016, at " \
                "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this " \
                "was the 50th Super Bowl, the league emphasized the \"golden anniversary\" " \
                "with various gold-themed initiatives, as well as temporarily suspending the " \
                "tradition of naming each Super Bowl game with Roman numerals (under which the game " \
                "would have been known as \"Super Bowl L\"), so that the logo could prominently " \
                "feature the Arabic numerals 50."
    import logging
    from logging.config import fileConfig
    fileConfig('logs/logging_config.ini')
    logger = logging.getLogger('logFile')
    NER(para_text, logger)
    batch = next(new_quest_data())
