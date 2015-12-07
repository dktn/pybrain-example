<!-- class: center, middle, inverse -->

# Neural Networks using PyBrain

---
## Short PyBrain tutorial

Installation:

```
$ git clone git://github.com/pybrain/pybrain.git

$ cd pybrain

$ sudo python setup.py install
```

---
## "Hello world!" in PyBrain

```
net = buildNetwork(2, 3, 1, bias=False, recurrent=False)

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (1,))
```

http://pybrain.org/docs/quickstart/network.html
http://pybrain.org/docs/quickstart/dataset.html

---
## Training example
```
trainer = BackpropTrainer(net, ds, learningrate=0.9, momentum=0.0, weightdecay=0.0, verbose=True)

trainer.trainEpochs(epochs=30)

print '0,0->', net.activate([0,0])
print '0,1->', net.activate([0,1])
print '1,0->', net.activate([1,0])
print '1,1->', net.activate([1,1])
```

http://pybrain.org/docs/quickstart/training.html

---
## buildNetwork

From PyBrain documentation:

**Layers** should be a list or tuple of integers, that indicate how many neurons the layers should have.

**Bias** and **outputbias** are flags to indicate whether the network should have the corresponding biases; both default to True.

To adjust the classes for the layers use the **hiddenclass** and **outclass** parameters, which expect a subclass of NeuronLayer.

If the **recurrent** flag is set, a RecurrentNetwork will be created, otherwise a FeedForwardNetwork.

If the **fast** flag is set, faster arac networks will be used instead of the pybrain implementations.

---
## BackpropTrainer class

From PyBrain documentation:

The **learningrate** gives the ratio of which parameters are changed into the direction of the gradient.

The learning rate decreases by **lrdecay**, which is used to to multiply the learning rate after each training step.

The parameters are also adjusted with respect to **momentum**, which is the ratio by which the gradient of the last timestep is used.

If **batchlearning** is set, the parameters are updated only at the end of each epoch. Default is False.

**Weightdecay** corresponds to the weightdecay rate, where 0 is no weight decay at all.

**Verbose** flag provides diagnostic output.

---
## Custom network construction

```
inLayer     = LinearLayer(2)
hiddenLayer = SigmoidLayer(4)
outLayer    = LinearLayer(1)

net = FeedForwardNetwork()

net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

in_to_hidden  = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

net.sortModules()
```

http://pybrain.org/docs/tutorial/netmodcon.html

---
## Run training

```
tst_ds, trn_ds = ds.splitWithProportion(0.25)

trainer.trainOnDataset(trn_ds, 10)

trainer.trainUntilConvergence(trn_ds, maxEpochs=100, verbose=True,
                continueEpochs=10, validationProportion=0.25)

trainer.testOnData(tst_ds, verbose=True)
```

http://pybrain.org/docs/tutorial/fnn.html
