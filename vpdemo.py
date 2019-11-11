import numpy as np
import keras
from tensorflow.contrib import learn
from keras import layers
from keras.models import Model

maxlen = 20
max_features = 2

trainSentences = [
    "that was great",
    "that movie is terrible",
    "that movie was bad",
    "that movie was terrible and it was really bad",
    "that was awesome and great",
    "wow that was so good it was great",
    "that was good",
    "wow you are awful"
]

trainLabel = [1, 0, 0, 0, 1, 1, 1, 0]

vp = learn.preprocessing.VocabularyProcessor(max_document_length=20, min_frequency=0, tokenizer_fn=list)

trainTokens = vp.fit_transform(trainSentences)

testSetences = [
    "wow how good was that",
    "i am feeling great",
    "it is terrible to feel bad",
    "i feel really awful today"
]

testLabel = [1, 1, 0, 0]

testTokens = vp.transform(testSetences)
print('trainTokens: {}'.format(np.array(list(trainTokens))))
print('testTokens: {}'.format(np.array(list(testTokens))))
