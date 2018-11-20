# The flow of this file is largely based on the work of @Currie32 on Github
# And also the public Github sharing with the link：
# https://github.com/Currie32/Chatbot-from-Movie-Dialogue/blob/master/Chatbot_Attention.ipynb
# Work from other Github contributor are also well documented in the following code
#---------------------------------------------------------------------------
# Environment: Ubuntu Desktop on Windows 10, Python 3.5, TensorFlow 1.0.0
# Please check the version of Python interpreter and TensorFlow module

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time

# Ignore system error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# PART A. Loading data
# ----------------------------------------------------------------------------------------
# Source data are publicaly available on the website: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
# Credit to Cristian Danescu-Niculescu-Mizil and Lillian Lee, Stanford University(Year 2011)
# Major part of the coding for data loading and inspection is credit to Github contributor suriyadeepan
# with link： https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/data.py
# Obtain sets of input data containing text with specified length


lines = open('./data/movie_lines.txt', encoding='utf-8',errors='ignore').read().split('\n')
conversation_lines = open('./data/movie_conversations.txt',encoding = 'utf-8',errors='ignore').read().split('\n')

# Create dictionary to map id for each line with its text
id2line = {}
for line in lines:
        line_ = line.split(' +++$+++ ')
        if len(line_) == 5:
                id2line[line_[0]] = line_[4]

# Create  list of all conversation lines ids
conversation = []
for line in conversation_lines[:-1]:
        line_ = line.split(' +++$+++ ')[-1][1:-1].replace("'","")
        line_ = line_.replace(" ", "")
        conversation.append(line_.split(','))

# Rearrange the lines to questions and answers
questions = []
answers = []

for line in conversation:
        if len(line) % 2 != 0:
                line = line [:-1]
        for i in range(len(line)):
                if i % 2 == 0:
                        questions.append(id2line[line[i]])
                else:
                        answers.append(id2line[line[i]])

# Further clean the data to remove redundant characters and reformat
# the wordings of the lines
def clean_text(text):
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
        return text

# Clean the questions and answers:
cleaned_questions, cleaned_answers = [], []
for question in questions:
        cleaned_questions.append(clean_text(question))
for answer in answers:
        cleaned_answers.append(clean_text(answer))


# Then we find the length of the lines:
lengths = []
for question in cleaned_questions:
        lengths.append(len(question.split()))
for answer in cleaned_answers:
        lengths.append(len(answer.split()))
lengths = pd.DataFrame(lengths, columns = ['counts'])

# Further make the scale smaller by removing the questions and answers shorter
# than 2 and longer than 20 characters, which makes up a large part of all data
selected_questions, selected_answers = [], []
i = 0
min_len = 2; max_len = 20
for i in range(len(cleaned_questions)-1):
        if min_len <= len(cleaned_questions[i].split()) <= max_len and \
                                min_len <= len(cleaned_answers[i].split()) <= max_len:
                selected_questions.append(cleaned_questions[i])
                selected_answers.append(cleaned_answers[i])

# Create a dictionary for frequency of the vocabulary
vocab = {}
for question in selected_questions:
        for word in question.split():
                if word not in vocab:
                        vocab[word] = 1
                else:
                        vocab[word] += 1

# Set threshold for some uncommon words, say show less than 2 times to keep the total
# number of the vocabulary used including less than 8% of UNK words
threshold = 2
count = 0
for k, v in vocab.items():
        if v >= threshold:
                count += 1


# Create dictionaries to provide unique integer for each word in
# the vocabulary used
question_vocab2int, answer_vocab2int = {}, {}
word_num = 0
for word, count in vocab.items():
        if count >= threshold:
                question_vocab2int[word] = word_num
                answer_vocab2int[word] = word_num
                word_num += 1

# Add unique tokens to the vocabulary dictionaries.
codes = ['<PAD>','<EOS>','<UNK>','<GO>']
for code in codes :
        question_vocab2int[code] = len(question_vocab2int) + 1
        answer_vocab2int[code] = len(answer_vocab2int) + 1

# Create accordingly the dictionaries to map integer to respective words
question_int2vocab = {i: v for v, i in question_vocab2int.items()}
answer_int2vocab = {i: v for v, i in answer_vocab2int.items()}

# Add token for end of sentence to the end of answers and
# then convert all text to integers
for i in range(len(selected_answers)):
        selected_answers[i] += '<EOS>'

questions_int, answers_int = [], []
for question in selected_questions:
        ints = []
        for word in question.split():
                if word not in question_vocab2int:
                        ints.append(question_vocab2int['<UNK>'])
                else:
                        ints.append(question_vocab2int[word])
        questions_int.append(ints)

for answer in selected_answers:
        ints = []
        for word in answer.split():
                if word not in answer_vocab2int:
                        ints.append(answer_vocab2int['<UNK>'])
                else:
                        ints.append(answer_vocab2int[word])
        answers_int.append(ints)

# Sort questions and answers by the length of the text
# To reduce the amount of padding during training
# Speed up training process with fewer loss

sorted_questions, sorted_answers = [], []
for length in range(1, max_len + 1):
        for i in enumerate(questions_int):
                if len(i[1]) == length:
                        sorted_questions.append(questions_int[i[0]])
                        sorted_answers.append(answers_int[i[0]])

						
# PART B.Seq2Seq Model Building
# -------------------------------------------------------------------------
# The following model part is from the work of Github user Currie32 with link
# https://github.com/Currie32/Chatbot-from-Movie-Dialogue/blob/master/Chatbot_Attention.ipynb

def model_inputs():
    # Create placeholders for inputs to the model
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob

def process_encoding_input(target_data, vocab_to_int, batch_size):
    # Remove the last word id from each batch and concat the <GO> to the begining of each batch
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    # Create the encoding layer
    lstm = tf.contrib.rnn.LSTMCell(num_units=rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,
                                                   cell_bw = enc_cell,
                                                   sequence_length = sequence_length,
                                                   inputs = rnn_inputs,
                                                   dtype=tf.float32)
    return enc_state				   

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob, batch_size):
    # Decode the training data
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = \
            tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",
                                                 num_units=dec_cell.output_size)
    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name = "attn_dec_train")
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                              train_decoder_fn,
                                                              dec_embed_input,
                                                              sequence_length,
                                                              scope=decoding_scope)
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    return output_fn(train_pred_drop)												 

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):
    # Decode the prediction data
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = \
            tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",
                                                 num_units=dec_cell.output_size)
    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn,
                                                                         encoder_state[0],
                                                                         att_keys,
                                                                         att_vals,
                                                                         att_score_fn,
                                                                         att_construct_fn,
                                                                         dec_embeddings,
                                                                         start_of_sequence_id,
                                                                         end_of_sequence_id,
                                                                         maximum_length,
                                                                         vocab_size,
                                                                         name = "attn_dec_inf")
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                                infer_decoder_fn,
                                                                scope=decoding_scope)

    return infer_logits

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size):
    # Create the decoding cell and input the parameters for the training and inference decoding layers

    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.LSTMCell(num_units=rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x,
                                                                vocab_size,
                                                                None,
                                                                scope=decoding_scope,
                                                                weights_initializer = weights,
                                                                biases_initializer = biases)                                                               weights_initializer$                                                                biases_initializer $
        train_logits = decoding_layer_train(encoder_state,
                                            dec_cell,
                                            dec_embed_input,
                                            sequence_length,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state,
                                            dec_cell,
                                            dec_embeddings,
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'],
                                            sequence_length - 1,
                                            vocab_size,
                                            decoding_scope,
                                            output_fn, keep_prob,
                                            batch_size)

    return train_logits, infer_logits

def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size,
                  questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers,
                  questions_vocab_to_int):
    # Use the previous functions to create the training and inference logits

    enc_embed_input = tf.contrib.layers.embed_sequence(input_data,
                                                       answers_vocab_size + 1,
                                                       enc_embedding_size,
                                                       initializer = tf.random_uniform_initializer(0,1))
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size+1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    train_logits, infer_logits = decoding_layer(dec_embed_input,
                                                dec_embeddings,
                                                enc_state,
                                                questions_vocab_size,
                                                sequence_length,
                                                rnn_size,
                                                num_layers,
                                                questions_vocab_to_int,
                                                keep_prob,
                                                batch_size)
    return train_logits, infer_logits


# PART C. Specific Features and Put Lines in Batches
# ----------------------------------------------------------------
# The following model part is from the work of Github user Currie32 with link
# https://github.com/Currie32/Chatbot-from-Movie-Dialogue/blob/master/Chatbot_Attention.ipynb

# Prepare hyperparameters for specific case
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75

# Reset the graph to ensure that it is ready for training
tf.reset_default_graph()
# Start the session
sess = tf.InteractiveSession()

# Load the model inputs
input_data, targets, lr, keep_prob = model_inputs()
# Sequence length will be the max line length for each batch
sequence_length = tf.placeholder_with_default(max_len, None, name='sequence_length')
# Find the shape of the input data for sequence_loss
input_shape = tf.shape(input_data)

# Create the training and inference logits
train_logits, inference_logits = seq2seq_model(
    tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(answer_vocab2int),
    len(question_vocab2int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers,
    question_vocab2int)

# Create a tensor for the inference logits, needed if loading a checkpoint version of the model
tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

def pad_sentence_batch(sentence_batch, vocab_to_int):
    # Pad sentences with <PAD> so that each sentence of a batch has the same length
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def batch_data(questions, answers, batch_size):
    # Batch questions and answers together
    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, question_vocab2int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answer_vocab2int))
        yield pad_questions_batch, pad_answers_batch
# Validate the training with 10% of the data
train_valid_split = int(len(sorted_questions)*0.15)

# Split the questions and answers into training and validating data
train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_answers[train_valid_split:]

valid_questions = sorted_questions[:train_valid_split]
valid_answers = sorted_answers[:train_valid_split]




def train():
    display_step = 100
    stop_early = 0
    stop = 3  # If the validation loss does decrease in 5 consecutive checks, stop training
    validation_check = ((len(train_questions)) // batch_size // 2) - 1  # Modulus for checking validation loss
    total_train_loss = 0  # Record the training loss for each display step
    summary_valid_loss = []  # Record the validation loss for saving improvements in the model

    checkpoint = "best_model.ckpt"  # Create a checkpoint file to keep track of training process
    sess.run(tf.global_variables_initializer())
	
    for epoch_i in range(1, epochs+1):
        for batch_i, (questions_batch, answers_batch) in enumerate(
				batch_data(train_questions, train_answers, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
				[train_op, cost],
				{input_data: questions_batch,
				 targets: answers_batch,
				 lr: learning_rate,
				 sequence_length: answers_batch.shape[1],
				 keep_prob: keep_probability})

            total_train_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
					  .format(epoch_i,
							  epochs,
							  batch_i,
							  len(train_questions) // batch_size,
							  total_train_loss / display_step,
							  batch_time*display_step))
                total_train_loss = 0

            if batch_i % validation_check == 0 and batch_i > 0:
                total_valid_loss = 0
                start_time = time.time()
                for batch_ii, (questions_batch, answers_batch) in \
						enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                    valid_loss = sess.run(
					cost, {input_data: questions_batch,
						   targets: answers_batch,
						   lr: learning_rate,
						   sequence_length: answers_batch.shape[1],
						   keep_prob: 1})
                    total_valid_loss += valid_loss
                end_time = time.time()
                batch_time = end_time - start_time
                avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
                print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))
				
                #  Reduce learning rate, but not below its minimum value
                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
					learning_rate = min_learning_rate
				summary_valid_loss.append(avg_valid_loss)
                if avg_valid_loss <= min(summary_valid_loss):
					print('New improvements!')
					stop_early = 0
					saver = tf.train.Saver()
					saver.save(sess, checkpoint) 
				else:
					print("No Improvement.")
					stop_early += 1
					if stop_early == stop:
						break

		if stop_early == stop:
			print("Stopping Training.")
			builder = tf.saved_model.builder.SavedModelBuilder("/output/rnn_model")
			break

# Part D. Model Testing
# ---------------------------------------------------------
# Test the Model Performance by input random input questions
# Reverse the question to integers and run repeatedly
def question_to_seq(question, vocab_to_int):
    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]

def predict():
    while True:
        # Allow input from users
        # input_question_raw = input("User:")
        # Here set to randomly generate questions from question base
        random = np.random.choice(len(selected_questions))
        input_question = selected_questions[random]
        input_question = question_to_seq(input_question, question_vocab2int)
        input_question = input_question + [question_vocab2int["<PAD>"]] * (max_len - len(input_question))
        # Add empty questions so the the input_data is the correct shape
        batch_shell = np.zeros((batch_size, max_len))
        # Set the first question to be out input question
        batch_shell[0] = input_question

        # Run the model with the input question
        answer_logits = sess.run(inference_logits, {input_data: batch_shell,
                                                    keep_prob: 1.0})[0]
        # Remove the padding from the Question and Answer
        pad_q = question_vocab2int["<PAD>"]
        pad_a = answer_vocab2int["<PAD>"]
        print('Bot: {}'.format([answer_int2vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))


# Select the mode for the program(train/predict)
# Demo to call the program in terminal: python chatbot.py train
if sys.argv[1] == 'train':
    train()
elif sys.argv[1] != 'train':
    sys.exit('Sorry we don\'t understand the request.')
else:
    predict()