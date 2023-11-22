# Replica

Replica is a data generation tool that leverages several machine learning architectures, including a ground-up implementation of the GPT-2 model, to generate data similar to data that is passed to the application as input.

### Inspiration

I always thought ML applications suffered especially during their training procedures. The time it took to train, data required to train, and environment were all hinderances to ML models. So I thought to create a model that can infinitely generate data similar to the data you give it. This way, you remove the obstacle of always having to find data to train your model with.

FYI: It does not accept all types of data (i.e mp3 files, images, pixels, etc)

### Topics

- Languages: Python
- Frameworks/Libraries: PyTorch
- Architectures: Transformers, RNNs/CNNs/GRUs, Bigrams, BoWs, GPT

### Use It Yourself

It is as simple as cloning, installing the right python dependencies as prompted, and running the Python file in any IDE with the right interpreter.

The Juypter Notebooks follow the above, just run the entire file or any cell given the right IDE and environment.

### Architectures (In Detail)

It is for those who are are curious/interested, in the underlying architecture of each of the models used in Replica.

#### Bigram

A bigram language model is a statistical language model used in natural language processing (NLP) and computational linguistics to predict the likelihood of a word based on the preceding word. It is a type of n-gram model where "n" is set to 2, meaning it considers pairs of consecutive words in a text.

In a bigram language model, the probability of a word is calculated based on the occurrence and frequency of word pairs in a training corpus. Specifically, it estimates the probability of a word "w" occurring after a preceding word "v."

In Replica, the Bigram Language Model, essentially a 'neural net' in structure, is simply a lookup table of logits for the
next character given a previous character, and it follows that in implementation.

Bigram Diagram:

<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/Bigram.PNG" />

#### Bag of Words (BoW)

The Bag of Words (BoW) model is a simple and fundamental language representation technique used in natural language processing (NLP) and text analysis. It's not a language model in the sense of a neural network-based language model but rather a basic method for representing and analyzing text data. BoW is used for various NLP tasks, such as document classification, sentiment analysis, and information retrieval.

Here's the general procedure:

- Tokenization
- Vocabulary Creation
- Vectorization
- Sparse Representation
- Vector Space Model (VSM)

A following example:

<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/BoW.PNG" />

BoW is straightforward and easy to implement, making it a good starting point for text analysis. However, it has several limitations:

- Loss of Word Order
- Lack of Semantics
- High Dimensionality
- Sparse Data

To address some of these limitations, more advanced text representation methods like TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings (e.g., Word2Vec, GloVe) have been developed. These methods aim to capture more of the semantics and context of words in text data.

#### RNNs/CNNs/GRUs/LSTMs

Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), Gated Recurrent Units (GRUs), and Long Short-Term Memory (LSTM) networks are all types of neural network architectures used in deep learning for various tasks, including natural language processing, image analysis, and sequential data modeling. Here's an overview of each and their architectures:

Recurrent Neural Networks (RNNs):
Architecture: RNNs are designed for processing sequential data. They have a simple architecture with one or more recurrent layers where each neuron in the layer maintains a hidden state that is updated at each time step. RNNs can be unidirectional (information flows from the past to the future) or bidirectional (information flows in both directions).
How they work: RNNs process sequences one element at a time, and the hidden state at each time step depends on the input at that time step and the previous hidden state. This enables them to capture dependencies within sequential data. However, standard RNNs have problems with vanishing and exploding gradients.

RNN Diagram:

<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/RNN.PNG" />

Convolutional Neural Networks (CNNs):
Architecture: CNNs are primarily used for image analysis and have a layered architecture consisting of convolutional layers, pooling layers, and fully connected layers. Convolutional layers use filters (kernels) to capture local patterns in the input data.
How they work: CNNs apply convolution operations to the input data to extract local features and use pooling layers to reduce spatial dimensions while preserving important information. They are known for their ability to capture spatial hierarchies and invariances in data, making them highly effective in image classification and other visual tasks.

Gated Recurrent Units (GRUs):
Architecture: GRUs are a type of RNN variant designed to address some of the issues of standard RNNs. They consist of two gates, an update gate and a reset gate, in addition to the hidden state. These gates control the information flow and enable GRUs to capture long-term dependencies more effectively.
How they work: The update gate controls how much of the previous hidden state should be combined with the current input, and the reset gate controls how much of the previous hidden state should be forgotten. GRUs can maintain information over longer sequences without suffering from vanishing gradients to the same extent as traditional RNNs.

GRU Diagrams:

<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/GRUs_1.PNG" />
<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/GRUs_2.PNG" />

Long Short-Term Memory (LSTM) networks:
Architecture: LSTMs are another RNN variant with a more complex architecture compared to traditional RNNs. They have three gates: input gate, forget gate, and output gate, which control the flow of information through the network. LSTMs also have a cell state in addition to the hidden state.
How they work: LSTMs are capable of capturing long-term dependencies in sequences and mitigating the vanishing gradient problem by regulating the flow of information. The input gate controls what information to store in the cell state, the forget gate controls what to forget, and the output gate controls what information to output from the cell state.

LSTM Diagrams:

<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/LSTM_1.PNG" />
<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/LSTM_2.PNG" />

Similarities:

- RNNs, GRUs, and LSTMs are all designed for sequential data and can capture dependencies over time.
- GRUs and LSTMs address the vanishing gradient problem better than traditional RNNs.
- CNNs and RNN variants (e.g., CNN-LSTM) are often used together in tasks involving both spatial and temporal data.

The fundamental structure of Replica advances from a simple NLP solution to a structure that leverages Deep Neural Networks (DNNs) like the ones mentioned above. We eventually move from this architecture to an advanced CNN, aka our WaveNet.

Replica Implementation:
I implemented the RNN and GRU cells as well as their respective modules. No implementation of LSTM but it nearly functions the same and is structured almost the same.

#### Transformers

A Transformer is a type of deep learning model architecture that has had a significant impact on natural language processing (NLP) and various other machine learning tasks. It was introduced in the paper "Attention is All You Need" (as I referenced to) by Vaswani et al. in 2017 and has since become a fundamental building block for many state-of-the-art NLP models, such as BERT, GPT, and more.

The transformer architecture in Replica is near equivalent in implementation stratetgy as the infamous chatGPT, particularly GPT 2.0. The activation function used is a Gaussian Error Linear Unit (GELU) which is typically used in deep neural networks. It was introduced as an alternative to traditional activation functions like ReLU (Rectified Linear Unit) and it is known for its smoothness and performance in certain applications.

I also create a vanilla multi-head masked self-attention layer with a projection at the end as part of the structure, along with the transformer block and the language model, aka our GPT2.0.

Transformer Diagrams:

<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/Transformer_1.PNG" />
<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/Transformer_2.PNG" />

#### WaveNet

WaveNet is a deep generative model for generating audio waveforms, developed by researchers at DeepMind, a subsidiary of Alphabet Inc. It was introduced in a paper titled "WaveNet: A Generative Model for Raw Audio" (as I referenced to) in 2016. WaveNet is known for its ability to generate highly realistic and natural-sounding speech and music.

WaveNet's architecture is based on deep neural networks, specifically deep convolutional neural networks (CNNs) and autoregressive models.

We use the hierarchical model to predict the next set of letters given a previous set of letters.

For example:

```
........ --> y
.......y --> u
......yu --> h
.....yuh --> e
....yuhe --> n
...yuhen --> g
..yuheng --> .
........ --> d
.......d --> i
......di --> o
.....dio --> n
....dion --> d
...diond --> r
..diondr --> e
.diondre --> .
........ --> x
.......x --> a
......xa --> v
.....xav --> i
....xavi --> e
```

It is like creating lego building blocks that connect together in such a way where you can feed data between said layers and receive output because of the hierarchical model.

WaveNet Diagram:

<img src="https://github.com/ReshiAdavan/Replica/blob/master/imgs/WaveNet.PNG" />

If you made it this far, congrats! That concludes Replica's README.
