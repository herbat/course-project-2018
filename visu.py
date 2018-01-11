import pickle
import numpy as np
import matplotlib.pyplot as plt


dict = pickle.load(open('save.p', 'rb'))
weights = dict['W1']

neuron1 = weights[101]
neuron1 = neuron1.reshape(80, 80)
print(np.shape(neuron1))
plt.imshow(neuron1)