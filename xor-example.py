from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer

net = buildNetwork(2, 3, 1, bias=True)
ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

# trainer = BackpropTrainer(net, ds)
# trainer.trainUntilConvergence()

trainer = BackpropTrainer(net, ds, learningrate = 0.9, momentum=0.0, weightdecay=0.0, verbose=True)
trainer.trainEpochs(epochs=300)

print '0,0->', net.activate([0,0])
print '0,1->', net.activate([0,1])
print '1,0->', net.activate([1,0])
print '1,1->', net.activate([1,1])
