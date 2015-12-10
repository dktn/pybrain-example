<!-- class: center, middle, inverse -->

# Neural Networks using PyBrain

---
## Short PyBrain guide

#### Adam Szlachta

contact: [adamsz@agh.edu.pl](mailto:adamsz@agh.edu.pl)

#### PyBrain usage examples:
```Bash
$ git clone https://github.com/dktn/pybrain-example.git
```

#### Links

- [http://pybrain.org/docs/](http://pybrain.org/docs/)
- [https://github.com/pybrain/pybrain/downloads](https://github.com/pybrain/pybrain/downloads)
- UCI Machine Learning Repository [http://archive.ics.uci.edu/ml/](http://archive.ics.uci.edu/ml/)

---

## PyBrain installation:

#### Windows:

1. Python 2.7 installation hints: [http://docs.python-guide.org/en/latest/starting/install/win/](http://docs.python-guide.org/en/latest/starting/install/win/)

2. Instalation from compiled libraries (download files from: [http://www.lfd.uci.edu/~gohlke/pythonlibs/](http://www.lfd.uci.edu/~gohlke/pythonlibs/))

```Bash
pip install "numpy‑1.9.3+mkl‑cp27‑none‑win32.whl"

pip install "scipy‑0.16.1‑cp27‑none‑win32.whl"

pip install "matplotlib‑1.5.0‑cp27‑none‑win32.whl"

pip install pybrain
```
---

## PyBrain installation:

#### Linux / Mac

```Bash
$ sudo zypper install python-numpy python-scipy python-matplotlib  # for Linux (OpenSuse)

$ brew install numpy scipy matplotlib  # for Mac

$ git clone git://github.com/pybrain/pybrain.git

$ cd pybrain

$ sudo python setup.py install
```

---
## "Hello world!" in PyBrain

```Python
net = buildNetwork(2, 3, 1, bias=False, recurrent=False, hiddenclass=SigmoidLayer)

# 2 - inputs
# 3 - neurons in 1st layer
# 1 - neurons in output layer

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

# (0,0)  - input
# (0,)   - expected output
```

[http://pybrain.org/docs/quickstart/network.html](http://pybrain.org/docs/quickstart/network.html)

[http://pybrain.org/docs/quickstart/dataset.html](http://pybrain.org/docs/quickstart/dataset.html)

---
## PyBrain documentation

```Python
net = buildNetwork(2, 3, 4, 1, bias=False, recurrent=False, hiddenclass=TanhLayer)
```

**Layers** should be a list or tuple of integers, that indicate how many neurons the layers should have.

**Bias** and **outputbias** are flags to indicate whether the network should have the corresponding biases; both default to True.

To adjust the classes for the layers use the **hiddenclass** and **outclass** parameters, which expect a subclass of NeuronLayer.

If the **recurrent** flag is set, a RecurrentNetwork will be created, otherwise a FeedForwardNetwork.

If the **fast** flag is set, faster arac networks will be used instead of the pybrain implementations.

---
## Training example
```Python
trainer = BackpropTrainer(net, ds, learningrate=0.9, momentum=0.0,
                          weightdecay=0.0, verbose=True)

trainer.trainEpochs(epochs=300)

print '0,0->', net.activate([0,0])
print '0,1->', net.activate([0,1])
print '1,0->', net.activate([1,0])
print '1,1->', net.activate([1,1])
```

[http://pybrain.org/docs/quickstart/training.html](http://pybrain.org/docs/quickstart/training.html)

---
## PyBrain documentation

```Python
trainer = BackpropTrainer(net, ds, learningrate=0.1, momentum=0.4,
                          weightdecay=0.0, batchlearning=False, verbose=True)
```

The **learningrate** gives the ratio of which parameters are changed into the direction of the gradient.

The learning rate decreases by **lrdecay**, which is used to to multiply the learning rate after each training step.

The parameters are also adjusted with respect to **momentum**, which is the ratio by which the gradient of the last timestep is used.

If **batchlearning** is set, the parameters are updated only at the end of each epoch. Default is False.

**Weightdecay** corresponds to the weightdecay rate, where 0 is no weight decay at all.

**Verbose** flag provides diagnostic output.

---
## Custom network construction

```Python
input_layer  = LinearLayer (6, name='in')
hidden_layer = SigmoidLayer(8, name='hid')
output_layer = LinearLayer (1, name='out')
bias_unit    = BiasUnit    (name='bias')

input_to_hidden  = FullConnection(input_layer,  hidden_layer)
hidden_to_output = FullConnection(hidden_layer, output_layer)
bias_to_hidden   = FullConnection(bias_unit,    hidden_layer)
bias_to_output   = FullConnection(bias_unit,    output_layer)

net = FeedForwardNetwork()

net.addInputModule (input_layer)
net.addModule      (hidden_layer)
net.addModule      (bias_unit)
net.addOutputModule(output_layer)

net.addConnection(input_to_hidden)
net.addConnection(hidden_to_output)
net.addConnection(bias_to_hidden)
net.addConnection(bias_to_output)

net.sortModules()
```

[http://pybrain.org/docs/tutorial/netmodcon.html](http://pybrain.org/docs/tutorial/netmodcon.html)

---
## Run training

```Python
tst_ds, trn_ds = ds.splitWithProportion(0.25)

trainer.trainOnDataset(trn_ds, 10)

trainer.trainUntilConvergence(trn_ds, maxEpochs=100, verbose=True,
                              continueEpochs=10, validationProportion=0.25)

trainer.testOnData(tst_ds, verbose=True)
```

[http://pybrain.org/docs/tutorial/fnn.html](http://pybrain.org/docs/tutorial/fnn.html)
