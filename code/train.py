import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.layers.core import Dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm, trange
from sklearn.model_selection import KFold

from Preparing_Data import training_data, _MAX_BATCH_SIZE as batch_size
from embedding import glove

seed = 3
folds = 5
iter_num = 1
params_headers = ['lstm_units','batch_size','epochs','encoder_layers','dropout_keep','learning_rate']
params = [100,batch_size,24,2,0.8,0.0002]
params_ranges = [[]]


def optimize():
    np.random.seed(seed)
    data = list(training_data())
    for iter in range(iter_num):
        for i, arr in enumerate(params_ranges):
            results = np.zeros([len(arr), 1])
            for j, tmp in enumerate(arr):
                params[i] = tmp
                print('Start kfold on params {}'.format(params))
                results[j] = kfold(data, params)
            print('scores for {0}: {1}\nmean = {2}'.format(params_headers[i],results,results[j]))
            with open('./Results/Scores/{0}_scores_{1}.txt'.format(params_headers[i],tmp), 'w') as file:
                for k, value in enumerate(results):
                    file.write('result {0} = {1}\n'.format(k, value))
            params[i] = arr[np.argmin(results)]
        print('Iteration {0} finished\nThe selected parameters are:\n{1}'.format(iter, params))
    # Save params:
    with open('./Results/Params/params.txt','w') as file:
        for i, value in enumerate(params):
            file.write('{0} = {1}\n'.format(params_headers[i],value))
    return params


def kfold(data, params):
    results = []
    kf = KFold(n_splits=folds, random_state=seed)
    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        validate_data = [data[i] for i in test_index]
        print('Start training')
        results.append(train(train_data, validate_data, params))
        print('Done validate\nresults = {}'.format(results))
        tf.reset_default_graph()

    return np.mean(results)


def train(train_data, validate_data, params):
    train_data_gen = (n for n in train_data)    # Convert the list to generator
    params_dict = {'lstm_units': params[0], "batch_size": params[1], 'epochs': params[2], 'num_encoder_layers': params[3], 'dropout': params[4], 'learning_rate': params[5]}
    graph = build_train_graph(params_dict)
    loss = graph["loss"]
    optimizer = tf.train.AdamOptimizer(learning_rate=params_dict['learning_rate']).minimize(loss)
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    epoch = 0
    batch_index = 0
    batch_count = None
    for epoch in trange(epoch + 1, params_dict['epochs'] + 1, desc="Epochs", unit="epoch"):
        batches = tqdm(train_data_gen, total=batch_count, desc="Batches", unit="batch")
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
        train_data_gen = (n for n in train_data)    # reset generator

    print('Done training\nStart validate')

    validate_data = (n for n in validate_data)  # Convert the list to generator
    batches_scores = []
    for i, batch in enumerate(validate_data):
        test_loss = session.run(loss, {
            graph['document_tokens']: batch["document_tokens"],
            graph['answer_masks']: batch["answer_masks"],
            graph['encoder_lengths']: batch["answer_lengths"],
            graph['decoder_inputs']: batch["question_input_tokens"],
            graph['decoder_labels']: batch["question_output_tokens"],
            graph['decoder_lengths']: batch["question_lengths"],
            graph['keep_prob']: params_dict["dropout"],
        })
        batches_scores.append(test_loss)

    session.close()
    return np.mean(batches_scores)


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

    training_helper = tf.contrib.seq2seq.TrainingHelper(
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
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units=params_dict["lstm_units"] * 2,
        memory=encoder_final_output,
        memory_sequence_length=encoder_lengths)

    # Building the Decoder
    temp_cell = LSTMCell(params_dict["lstm_units"] * 2, forget_bias=1.0)
    temp_cell = DropoutWrapper(temp_cell, output_keep_prob=keep_prob, )
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell=temp_cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=params_dict["lstm_units"] * 2)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=decoder_cell,
        helper=training_helper,
        initial_state=decoder_cell.zero_state(params_dict["batch_size"], tf.float32).clone(cell_state=encoder_final_state),
        output_layer=projection)

    training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=training_decoder,
        impute_finished=True,
        maximum_iterations=tf.reduce_max(decoder_lengths))

    training_logits = training_decoder_output.rnn_output
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
        "loss": loss
    }


if __name__ == "__main__":
    parameters = optimize()
