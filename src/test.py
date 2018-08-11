import json as js
import os
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.layers.core import Dense
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.scores import f_measure
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm, trange
from Preparing_Data import training_data, test_data, _MAX_BATCH_SIZE as batch_size
from embedding import glove, look_up_word, look_up_token, UNKNOWN_TOKEN, END_TOKEN

counter = 0
seed = 3
params_headers = ['lstm_units', 'batch_size', 'epochs', 'encoder_layers', 'dropout_keep', 'learning_rate']
params = [100, batch_size, 24, 2, 0.8, 0.0002]


def build_train_graph(params_dict):
    # Building the Embedding layer + placeholders
    keep_prob = tf.placeholder(tf.float32)
    embedding = tf.get_variable("embedding", initializer=glove, trainable=True)
    document_tokens = tf.placeholder(tf.int32, shape=[None, None], name="document_tokens")
    document_emb = tf.nn.embedding_lookup(embedding, document_tokens)
    answer_masks = tf.placeholder(tf.float32, shape=[None, None, None], name="answer_masks")
    decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inputs")
    decoder_labels = tf.placeholder(tf.int32, shape=[None, None], name="decoder_labels")
    decoder_lengths = tf.placeholder(tf.int32, shape=[None], name="decoder_lengths")
    encoder_lengths = tf.placeholder(tf.int32, shape=[None], name="encoder_lengths")
    decoder_emb = tf.nn.embedding_lookup(embedding, decoder_inputs)
    question_mask = tf.sequence_mask(decoder_lengths, dtype=tf.float32)
    projection = Dense(embedding.shape[0], use_bias=False)

    training_helper = seq2seq.TrainingHelper(
        inputs=decoder_emb,
        sequence_length=decoder_lengths,
        time_major=False)

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

    training_decoder = seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=training_helper,
        initial_state=decoder_cell.zero_state(params_dict["batch_size"], tf.float32).clone(cell_state=encoder_final_state),
        output_layer=projection)

    training_decoder_output, _, _ = seq2seq.dynamic_decode(
        decoder=training_decoder,
        impute_finished=True,
        maximum_iterations=tf.reduce_max(decoder_lengths))

    training_logits = training_decoder_output.rnn_output
    # Normalize the logits between [0,1]
    prob_logits = tf.nn.softmax(training_logits, axis=-1)

    loss = seq2seq.sequence_loss(logits=training_logits, targets=decoder_labels, weights=question_mask, name="loss")

    return {
        "keep_prob": keep_prob,
        "document_tokens": document_tokens,
        "answer_masks": answer_masks,
        "encoder_lengths": encoder_lengths,
        "decoder_inputs": decoder_inputs,
        "decoder_labels": decoder_labels,
        "decoder_lengths": decoder_lengths,
        "training_logits": training_logits,
        "prob_logits": prob_logits,
        "loss": loss
    }


def test(params):
    params_dict = {'lstm_units': params[0], "batch_size": params[1], 'epochs': params[2], 'num_encoder_layers': params[3], 'dropout': params[4], 'learning_rate': params[5]}
    graph = build_train_graph(params_dict)
    loss = graph["loss"]
    optimizer = tf.train.AdamOptimizer(learning_rate=params_dict['learning_rate']).minimize(loss)
    session = tf.Session()
    saver = tf.train.Saver()

    # Run only if we want to train the model over all the data again (in case we change the model)
    if not os.path.isfile('./Trained Model/model.meta'):
        print('start training')
        session.run(tf.global_variables_initializer())
        epoch = 0
        batch_index = 0
        batch_count = None
        for epoch in trange(epoch + 1, params_dict['epochs'] + 1, desc="Epochs", unit="epoch"):
            batches = tqdm(training_data(), total=batch_count, desc="Batches", unit="batch")
            for batch in batches:
                _, loss_value = session.run([optimizer, loss], {
                    graph['document_tokens']: batch["document_tokens"],
                    graph['answer_masks']: batch["answer_masks"],
                    graph['encoder_lengths']: batch["answer_lengths"],
                    graph['decoder_inputs']: batch["question_input_tokens"],
                    graph['decoder_labels']: batch["question_output_tokens"],
                    graph['decoder_lengths']: batch["question_lengths"],
                    graph['keep_prob']: params_dict["dropout"],
                })
                batches.set_postfix(loss=loss_value)
                batch_index += 1
            if batch_count is None:
                batch_count = batch_index

        saver.save(session, 'Trained Model/model')
    else:
        saver.restore(session, 'Trained Model/model')
        print('loaded model')

    print('Done training\nStart testing')
    # calculate loss to check negative log likelihood
    nll_scores = []
    for i, batch in enumerate(test_data()):
        test_loss = session.run(loss, {
            graph['document_tokens']: batch["document_tokens"],
            graph['answer_masks']: batch["answer_masks"],
            graph['encoder_lengths']: batch["answer_lengths"],
            graph['decoder_inputs']: batch["question_input_tokens"],
            graph['decoder_labels']: batch["question_output_tokens"],
            graph['decoder_lengths']: batch["question_lengths"],
            graph['keep_prob']: params_dict["dropout"],
        })
        nll_scores.append(float(test_loss))
    print('negative log likelihood check is Done')

    # Generate questions to check BLEU
    prob_logits = graph['prob_logits']
    bleu_scores = []
    fmeasure_scores = []
    for i, batch in enumerate(test_data()):
        questions = session.run(prob_logits, {
            graph['document_tokens']: batch["document_tokens"],
            graph['answer_masks']: batch["answer_masks"],
            graph['encoder_lengths']: batch["answer_lengths"],
            graph['decoder_inputs']: batch["question_input_tokens"],
            graph['decoder_labels']: batch["question_output_tokens"],
            graph['decoder_lengths']: batch["question_lengths"],
            graph['keep_prob']: params_dict["dropout"],
        })
        questions = paragraph_bonus(questions, batch["document_tokens"])
        questions[:, :, UNKNOWN_TOKEN] = 0
        questions = np.argmax(questions, 2)
        questions_list = []
        for i, quest in enumerate(questions):
            question = itertools.takewhile(lambda t: t != END_TOKEN, questions[i])
            generated_question = " ".join(look_up_token(token) for token in question)
            questions_list.append(generated_question)

        save_questions(questions_list, batch)
        bleu_scores.append(bleu(questions_list, batch))
        fmeasure_scores.append(fmeasure(questions_list, batch))
    print('BLEU and F_measure checks are Done')

    session.close()
    return nll_scores, bleu_scores, fmeasure_scores


def paragraph_bonus(questions, paragraph_tokens):
    stop_words = ['the', 'in', 'is', 'of', 'and', 'to', 'a']
    stop_words_index = [look_up_word(i) for i in stop_words]
    for i, example in enumerate(questions):
        for j, question in enumerate(example):
            for k, word in enumerate(set(paragraph_tokens[i])):
                if word not in stop_words_index:
                    questions[i, j, word] += 0.2
    return questions


def save_questions(questions_list, batch):
    global counter
    with open('Results/Generated Questions/TestQuest_batch_{}.txt'.format(counter), 'w+') as f:
      for i in range(batch["size"]):
          f.write('Paragraph: ' + batch["document_text"][i] + '\n')
          f.write("Question: " + questions_list[i] + '\n')
          f.write("Right question: " + batch['question_text'][i] + '\n')
          f.write("Answer: " + batch["answer_text"][i] + '\n\n')
    counter = counter + 1


def bleu(questions_list, batch):
    bleu_scores = []
    chencherry = SmoothingFunction()

    for i, question in enumerate(questions_list):
        original_quest = batch['question_text'][i]
        ref_quest = original_quest[:-1].split(' ')
        gen_quest = question.split(' ')
        score = sentence_bleu([ref_quest], gen_quest, smoothing_function=chencherry.method4)
        bleu_scores.append(float(score))

    return bleu_scores


def fmeasure(questions_list, batch):
    f_scores = []
    for i, question in enumerate(questions_list):
        original_quest = batch['question_text'][i]
        ref_quest = set(original_quest[:-1].split(' '))
        gen_quest = set(question.split(' '))
        score = f_measure(ref_quest, gen_quest)
        f_scores.append(float(score))

    return f_scores


def save_scores(nll_scores, bleu_scores, fmeasure_scores):
    nll_mean = np.mean(nll_scores)
    bleu_mean = np.mean(bleu_scores)
    fmeasure_mean = np.mean(fmeasure_scores)
    print(nll_mean)
    print(bleu_mean)
    print(fmeasure_mean)

    scores_json = {'Mean NLL': float(nll_mean),
                   'Mean BLEU': float(bleu_mean),
                   'Mean Fmeasure': float(fmeasure_mean)}
    for i in range(len(nll_scores)):
        scores_json['Batch {}'.format(i)] = {'NLL': nll_scores[i],
                                             'BLEU': bleu_scores[i],
                                             'Fmeasure': fmeasure_scores[i]}

    with open('Results/Scores/test_score.json', 'w') as f:
        js.dump(scores_json, f)
    #
    # with open('Results/Scores/test_score.txt', 'w') as res:
    #     res.write('Mean NLL test scores for all batches = {}\n'.format(nll_mean))
    #     res.write('Mean BLEU test scores for all batches = {}\n'.format(bleu_mean))
    #     for i, score in enumerate(nll_scores):
    #         res.write('NLL score for batch {0} = {1}\n'.format(i, score))
    #     for i, score in enumerate(bleu_scores):
    #         res.write('BLEU score for batch {0} = {1}\n'.format(i, score))


if __name__ == "__main__":
    nll_scores, bleu_scores , fmeasure_scores = test(params)
    save_scores(nll_scores, bleu_scores, fmeasure_scores)
