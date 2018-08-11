import time
import json as js
import os
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.layers.core import Dense
import logging
from logging.config import fileConfig

from Preparing_Data import _MAX_BATCH_SIZE as batch_size
from embedding import look_up_token, look_up_word, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN, glove
from NER import new_quest_data, NER

fileConfig('logs/logging_config.ini')
logger = logging.getLogger('logFile')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input_text_path = 'Data/input_text.txt'
model_path = 'Trained Model/model'
res_text_path = 'Results/results.txt'
res_json_path = 'Results/results.json'
params_headers = ['lstm_units', 'batch_size', 'epochs', 'encoder_layers', 'dropout_keep', 'learning_rate']
params = [100, batch_size, 24, 2, 0.8, 0.0002]
LSTM_UNITS = 128


def build_inference_graph(params_dict):
    # Todo: Check if load the glove is faster than import it from embedding
    # glove = np.load('GloVe/glove.npy')

    # Building the Embedding layer + placeholders
    keep_prob = tf.placeholder(tf.float32)
    embedding = tf.get_variable("embedding", initializer=glove, trainable=True)
    document_tokens = tf.placeholder(tf.int32, shape=[None, None], name="document_tokens")
    document_emb = tf.nn.embedding_lookup(embedding, document_tokens)
    answer_masks = tf.placeholder(tf.float32, shape=[None, None, None], name="answer_masks")
    encoder_lengths = tf.placeholder(tf.int32, shape=[None], name="encoder_lengths")
    projection = Dense(embedding.shape[0], use_bias=False)

    helper = seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch_size],
                                                              START_TOKEN), END_TOKEN)

    # Building the Encoder
    encoder_inputs = tf.matmul(answer_masks, document_emb, name="encoder_inputs")

    output = encoder_inputs
    for n in range(params_dict["num_encoder_layers"]):
        cell_fw = LSTMCell(params_dict["lstm_units"], forget_bias=1.0, state_is_tuple=True)
        cell_bw = LSTMCell(params_dict["lstm_units"], forget_bias=1.0, state_is_tuple=True)
        cell_fw = DropoutWrapper(cell_fw, output_keep_prob=keep_prob, )
        cell_bw = DropoutWrapper(cell_bw, output_keep_prob=keep_prob, )

        state_fw = cell_fw.zero_state(params_dict["batch_size"], tf.float32)
        state_bw = cell_bw.zero_state(params_dict["batch_size"], tf.float32)

        (output_fw, output_bw), encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output,
                                                                                initial_state_fw=state_fw,
                                                                                initial_state_bw=state_bw,
                                                                                sequence_length=encoder_lengths,
                                                                                dtype=tf.float32,
                                                                                scope='encoder_rnn_' + str(n))
        output = tf.concat([output_fw, output_bw], axis=2)

    encoder_final_output = output
    encoder_state_c = tf.concat((encoder_state[0][0], encoder_state[1][0]), -1)
    encoder_state_h = tf.concat((encoder_state[0][1], encoder_state[1][1]), -1)
    encoder_final_state = LSTMStateTuple(encoder_state_c, encoder_state_h)

    # Attention mechanism
    attention_mechanism = seq2seq.LuongAttention(
        num_units=params_dict["lstm_units"] * 2,
        memory=encoder_final_output,
        memory_sequence_length=encoder_lengths)

    # Building the Decoder
    temp_cell = LSTMCell(params_dict["lstm_units"] * 2, forget_bias=1.0)
    temp_cell = DropoutWrapper(temp_cell, output_keep_prob=keep_prob, )
    decoder_cell = seq2seq.AttentionWrapper(
        cell=temp_cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=params_dict["lstm_units"] * 2)

    decoder = seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=helper,
        initial_state=decoder_cell.zero_state(params_dict["batch_size"], tf.float32).clone(cell_state=encoder_final_state),
        output_layer=projection)

    decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder, maximum_iterations=16)
    decoder_outputs = decoder_outputs.rnn_output

    # Normalize the logits between [0,1]
    prob_logits = tf.nn.softmax(decoder_outputs, axis=-1)

    return {
        "keep_prob": keep_prob,
        "document_tokens": document_tokens,
        "answer_masks": answer_masks,
        "encoder_lengths": encoder_lengths,
        "decoder_outputs": decoder_outputs,
        "prob_logits": prob_logits
    }


def generate_quest():
    params_dict = {'lstm_units': params[0], "batch_size": params[1], 'epochs': params[2],
                   'num_encoder_layers': params[3], 'dropout': params[4], 'learning_rate': params[5]}
    graph = build_inference_graph(params_dict)

    # Load the input batch:
    try:
        batch = next(new_quest_data())
    except:
        res_json = {"Paragraph": "No questions generated on this text", "Questions": [], "Answers": []}
        with open(res_json_path, 'w+') as f:
            js.dump(res_json, f)
        logger.warning('Could not extract answers. Therefor, no questions generated on this text')
        quit()

    saver = tf.train.Saver()
    session = tf.Session()
    saver.restore(session, model_path)

    prob_logits = graph['prob_logits']
    questions = session.run(prob_logits, {
        graph['document_tokens']: batch["document_tokens"],
        graph['answer_masks']: batch["answer_masks"],
        graph['encoder_lengths']: batch["answer_lengths"],
        graph['keep_prob']: params_dict["dropout"],
    })
    questions = paragraph_bonus(questions, batch["document_tokens"])
    questions[:, :, UNKNOWN_TOKEN] = 0
    questions = np.argmax(questions, 2)
    last_answer_text = ''
    with open(res_text_path, 'w+', encoding='utf-8', errors='replace') as f:
        for i in range(batch["size"]):
            if batch["answer_text"][i] == last_answer_text:
                break
            question = itertools.takewhile(lambda t: t != END_TOKEN, questions[i])
            generatedQuestion = " ".join(look_up_token(token) for token in question)
            f.write('Paragraph: ' + batch["document_text"][i] + '\n')
            f.write("Question: " + generatedQuestion + '\n')
            f.write("Answer: " + batch["answer_text"][i] + '\n')
            f.write("Synonyms: " + str(batch["synonyms"][i]) + '\n\n')
            last_answer_text = batch["answer_text"][i]
    # NEED TO SAVE AS JSON FILE:
    jsonarr = {"Paragraph": batch["document_text"][0]}

    answer_list = []
    question_list = []
    for i in range(batch["size"]):
        question = itertools.takewhile(lambda t: t != END_TOKEN, questions[i])
        generatedQuestion = " ".join(look_up_token(token) for token in question)
        question_list.append(generatedQuestion)
        answer_list.append(batch["answer_text"][i])

    jsonarr["Questions"] = question_list
    jsonarr["Answers"] = answer_list

    with open(res_json_path, 'w+') as f:
        js.dump(jsonarr, f)


def paragraph_bonus(questions, paragraph_tokens):
    stop_words = ['the', 'in', 'is', 'of', 'and', 'to', 'a']
    stop_words_index = [look_up_word(i) for i in stop_words]
    for i, example in enumerate(questions):
        for j, question in enumerate(example):
            for k, word in enumerate(set(paragraph_tokens[i])):
                if word not in stop_words_index:
                    questions[i, j, word] += 0.2
    return questions


if __name__ == '__main__':
    start_time = time.time()
    logger.info("Process start")
    with open(input_text_path) as textFile:
        text = textFile.read()
        logger.info("File read done")
        NER(text, logger)
        logger.info("Extract answers from the text done")

    logger.info("Start generate questions")
    generate_quest()
    logger.info("Generate questions done\nExecution time = {} seconds".format(time.time()-start_time))
