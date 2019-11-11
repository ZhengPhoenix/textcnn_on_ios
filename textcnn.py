import numpy as np
import keras
from tensorflow.contrib import learn
from keras import layers
from keras.models import Model
from keras.preprocessing import sequence

maxlen = 20

mapping = {
    "terrible": 1,
    "great": 2,
    "bad": 3,
    "good": 4,
    "awful": 5,
    "awesome": 6
}

trainSentences = [
    "that was great",
    "that movie is terrible",
    "that movie was bad",
    "that movie was terrible and it was really bad",
    "that was awesome and great",
    "wow that was so good it was great",
    "that was good",
    "wow you are awful",
    "that was good",
    "that was awesome"
]

trainLabel = [1, 0, 0, 0, 1, 1, 1, 0, 1, 1]

def tokenize(sent):
    words = sent.split(' ')
    tokenized = []
    for word in words:
        if mapping.get(word) != None:
            tokenized.append(mapping[word])
        else:
            tokenized.append(0)
    return tokenized


vp = learn.preprocessing.VocabularyProcessor(max_document_length=maxlen, min_frequency=0, tokenizer_fn=list)

tokens = vp.fit_transform(trainSentences)
max_features = np.max(np.array(list(tokens))) + 1
print("maxfeatures: {}".format(max_features))

testSetences = [
    "wow how good was that",
    "i am feeling great",
    "it is terrible to feel bad",
    "i feel really awful today"
]

testLabel = [1, 1, 0, 0]

trainTokens = vp.transform(trainSentences)
testTokens = vp.transform(testSetences)

# now construct network

print('Build model...')
input_text = layers.Input(shape=(maxlen, ), name='input_text')
reshape_1 = layers.Reshape(input_shape=(maxlen, ), target_shape=(maxlen, 1, ))(input_text)
emb = layers.Embedding(max_features, 64, dropout=0.2, name='embedding')(reshape_1)
reshape_2 = layers.Reshape(target_shape=(maxlen, 64, ))(emb)

filter_size = 2
conv1d = layers.Conv1D(filters=32,
                     kernel_size=filter_size,
                     strides=1,
                     padding='valid',
                     activation='relu',
                     kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                     bias_initializer=keras.initializers.constant(value=0.1),
                     name='conv_%d' % filter_size)(reshape_2)
maxpool1d = layers.MaxPool1D(pool_size=maxlen - filter_size + 1,
                            strides=1,
                            padding='valid',
                            name=('maxpool_%d' % filter_size))(conv1d)

dropout = layers.Dropout(0.5)(maxpool1d)
reshape_3 = layers.Reshape(target_shape=(32, ))(dropout)
dense = layers.Dense(1, name='dense')(reshape_3)

model = Model(inputs=[input_text], outputs=[dense])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# construct training and validate data
xTrain = np.array(list(trainTokens))
xTest = np.array(list(testTokens))

yTrain = np.array(trainLabel)
yTest = np.array(testLabel)

print('train shape: {}'.format(xTrain.shape))

# start training
print("Start training...")
batch_size = 2
model.fit(xTrain, yTrain, batch_size=2, epochs=10, validation_data=(xTest, yTest))
# get model accuracy
score, acc = model.evaluate(xTest, yTest, batch_size=batch_size)

print('Test score: {}'.format(score))
print('Test accuracy: {}'.format(acc))

# now try to predict
predictSentence = ["that was good", "that was great"]
tokenPredict = []
for sent in predictSentence:
    tokenPredict.append(tokenize(sent))

predictToken = np.array(tokenPredict)

print('predict result: {}'.format(model.predict(sequence.pad_sequences(np.array(list(predictToken)), maxlen=maxlen))))