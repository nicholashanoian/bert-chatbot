# fine tune bert for seq2seq task using cornell movie dialogue corpus


from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import numpy as np

import pickle as pkl
import random


# bert imports
import bert
from bert import optimization
from bert import tokenization
from bert import extract_features


BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"



flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")


flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")



flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")



# tpu parameters

# flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

# flags.DEFINE_string(
#     "tpu_zone", None,
#     "[Optional] GCE zone where the Cloud TPU is located in. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")

# flags.DEFINE_string(
#     "gcp_project", None,
#     "[Optional] Project name for the Cloud TPU-enabled project. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")

# flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

# flags.DEFINE_integer(
#     "num_tpu_cores", 8,
#     "Only used if `use_tpu` is True. Total number of TPU cores to use.")





def process_and_save_data(movie_lines="cornell-data/movie_lines.txt",
                 movie_conversations="cornell-data/movie_conversations.txt"):
    '''used to generate the conversations.pkl file. only run once if you don't have it.
    the files specified as default arguments can be found in the cornell movie database zip'''

    movie_lines_features = ["LineID", "Character", "Movie", "Name", "Line"]
    movie_lines = pd.read_csv(movie_lines, sep = "\+\+\+\$\+\+\+", engine = "python", index_col = False, names = movie_lines_features)

    # Using only the required columns, namely, "LineID" and "Line"
    movie_lines = movie_lines[["LineID", "Line"]]

    # Strip the space from "LineID" for further usage and change the datatype of "Line"
    movie_lines["LineID"] = movie_lines["LineID"].apply(str.strip)


    # In[5]:


    movie_conversations_features = ["Character1", "Character2", "Movie", "Conversation"]
    movie_conversations = pd.read_csv(movie_conversations, sep = "\+\+\+\$\+\+\+", engine = "python", index_col = False, names = movie_conversations_features)

    # Again using the required feature, "Conversation"
    movie_conversations = movie_conversations["Conversation"]


    # In[6]:


    # This instruction takes lot of time, run it only once.
    conversations = [[str(list(movie_lines.loc[movie_lines["LineID"] == u.strip().strip("'"), "Line"])[0]).strip() for u in c.strip().strip('[').strip(']').split(',')] for c in movie_conversations]

    with open("conversations.pkl", "wb") as handle:
        pkl.dump(conversations, handle)

def read_examples_from_pkl(filename):
    with open(filename, "rb") as handle:
        conversations = pkl.load(handle)
    
    n_samples = 200
    train_ratio = 0.8
    cut = int(n_samples * train_ratio)

    # take sample of all convos
    convo_samples = random.sample(conversations, n_samples)
    convo_samples = [[convo[0], convo[1]] for convo in convo_samples]
    convo_samples = pd.DataFrame(convo_samples, columns=['a', 'b'])

    train, test = train_test_split(convo_samples)

    # axis='columns' because we want to index into the columns for each row x
    unique_id = 20
    train_InputExamples = train.apply(lambda x: bert.extract_features.InputExample(
        unique_id=unique_id,
        text_a=x['a'],
        text_b=x['b']), axis='columns')
    test_InputExamples = test.apply(lambda x: bert.extract_features.InputExample(
        unique_id=unique_id,
        text_a=x[0],
        text_b=x[1]), axis='columns')
    return train_InputExamples, test_InputExamples



def create_tokenizer_from_hub_module(bert_model_hub):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])
      
    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)






# # Creating a model
# 
# Now that we've prepared our data, let's focus on building a model. `create_model` does just this below. First, it loads the BERT tf hub module again (this time to extract the computation graph). Next, it creates a single new layer that will be trained to adapt BERT to our sentiment task (i.e. classifying whether a movie review is positive or negative). This strategy of using a mostly trained model is called [fine-tuning](http://wiki.fast.ai/index.php/Fine_tuning).

def create_model(is_predicting, input_ids, input_mask, segment_ids, vocab, vocab_size):
    """Creates a classification model."""

    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True)
    
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)

    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    output_layer = bert_outputs["sequence_output"]
    
    batch_size = output_layer.shape[0]
    max_seq_length = output_layer.shape[1]
    hidden_size = output_layer.shape[2]
    

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
        "output_weights", [vocab_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [vocab_size], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        # add a max_seq length stack of bias so that we add the bias to each word distributoin
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_answer = tf.one_hot(input_ids, depth=vocab_size)


        predictions = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predictions, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_answer * log_probs, axis=-1)
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=input_ids, logits=logits)
        
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predictions, log_probs)



# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(vocab_list, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["input_type_ids"]
        # label_ids = features["label_ids"]
        vocab = vocab_list
        vocab_size = len(vocab_list)

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predictions, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, vocab, vocab_size)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)
            ## else:
                # return tf.estimator.EstimatorSpec(mode=mode,
                # loss=loss,
                # eval_metric_ops=eval_metrics)
        else:
            (predictions, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, vocab, vocab_size)

            predictions = {
                'probabilities': log_probs,
                'predictions': predictions
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn





def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    # create output dir
    tf.gfile.MakeDirs(FLAGS.output_dir)



    
    # This is a path to an uncased (all lowercase) version of BERT
    tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)



    MAX_SEQ_LENGTH = 128
    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    # Warmup is a period of time where hte learning rate 
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100



    
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))


    train_InputExamples, test_InputExamples = read_examples_from_pkl(FLAGS.data_dir + "/conversations.pkl")
    train_features = bert.extract_features.convert_examples_to_features(
        train_InputExamples, MAX_SEQ_LENGTH, tokenizer)
    test_features = bert.extract_features.convert_examples_to_features(
        test_InputExamples, MAX_SEQ_LENGTH, tokenizer)


    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    
    # Specify outpit directory and number of checkpoint steps to save
    # run_config = tf.estimator.RunConfig(
    #     model_dir=FLAGS.output_dir,
    #     save_summary_steps=SAVE_SUMMARY_STEPS,
    #     save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)


    model_fn = model_fn_builder(
        vocab_list=[v for k,v in tokenizer.vocab.items()],
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})

        

    
    if FLAGS.do_train:
        
        train_input_fn = bert.extract_features.input_fn_builder(
            features=train_features,
            seq_length=MAX_SEQ_LENGTH)

        # train the model
        print('Beginning Training!')
        current_time = datetime.now()
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Training took time ", datetime.now() - current_time)

    if FLAGS.do_predict:
        def getPrediction(in_sentences):
            unique_id=100
            input_examples = [
                bert.extract_features.InputExample(unique_id=unique_id, text_a=x, text_b = "")
                for x in in_sentences]
            input_features = bert.extract_features.convert_examples_to_features(
                input_examples, MAX_SEQ_LENGTH, tokenizer)
            predict_input_fn = bert.extract_features.input_fn_builder(
                features=input_features, seq_length=MAX_SEQ_LENGTH)
            predictions = estimator.predict(predict_input_fn)
            return [(sentence, prediction['predictions']) for sentence, prediction in zip(in_sentences, predictions)]

        pred_sentences = [
            "Hi, how are you?",
            "What is your name?",
            "I love you",
            "Did you see the Yankees game last night?",
            # "Who is your favorite actor?"
        ]
        predictions = getPrediction(pred_sentences)
        for qa in [x + ", " + " ".join(tokenizer.convert_ids_to_tokens(y)) for x,y in predictions]:
            print(qa + "\n")




if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
