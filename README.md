# Replica

Replica is a data generation tool that leverages several machine learning architectures, including a ground-up implementaiton of the GPT-2 model, to generate data similar to data that is passed to the application as input.

### Inspiration

I always thought ML applications suffered especially during their training procedures. The time it took to train, data required to train, and environment were all hinderances to ML models. So I thought to create a model that can infinitely generate data similar to the data you give it. This way, you remove the obstacle of always having to find data to train your model with.

FYI: It does not accept all types of data (i.e mp3 files, images, pixels, etc)

### Topics

- Languages: Python
- Frameworks/Libraries: PyTorch
- Architectures: Transformers, RNNs/CNNs/GRUs, Bigrams, BoWs, GPT
