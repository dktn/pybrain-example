from pybrain.tools.shortcuts     import buildNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure           import TanhLayer, SoftmaxLayer, LinearLayer, SigmoidLayer

# buildNetwork(*layers, **options)
# layers should be a list or tuple of integers, that indicate how many neurons the layers should have.
# bias and outputbias are flags to indicate whether the network should have the corresponding biases; both default to True.
# To adjust the classes for the layers use the hiddenclass and outclass parameters, which expect a subclass of NeuronLayer.
# If the recurrent flag is set, a RecurrentNetwork will be created, otherwise a FeedForwardNetwork.
# If the fast flag is set, faster arac networks will be used instead of the pybrain implementations.

net = buildNetwork(2, 1, 1, bias=False) # exercise: setup better structure by experimenting

# net = buildNetwork(2, 2, 1,    bias=True,  hiddenclass=SigmoidLayer)
# net = buildNetwork(2, 3, 4, 1, bias=True,  hiddenclass=TanhLayer)
# net = buildNetwork(2, 3, 1,    bias=False, hiddenclass=TanhLayer, outclass=SoftmaxLayer)

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

# class BackpropTrainer(module, dataset=None, learningrate=0.01, lrdecay=1.0, momentum=0.0,
#                       verbose=False, batchlearning=False, weightdecay=0.0)
# The learningrate gives the ratio of which parameters are changed into the direction of the gradient.
# The learning rate decreases by lrdecay, which is used to to multiply the learning rate after each training step.
# The parameters are also adjusted with respect to momentum, which is the ratio by which the gradient of the last timestep is used.
# If batchlearning is set, the parameters are updated only at the end of each epoch. Default is False.
# weightdecay corresponds to the weightdecay rate, where 0 is no weight decay at all.

trainer = BackpropTrainer(net, ds, learningrate=0.9, momentum=0.0, weightdecay=0.0, verbose=True)

trainer.trainEpochs(epochs=30)

# trainer.trainUntilConvergence()  # exercise: why it does not work?

print '0,0->', net.activate([0,0])
print '0,1->', net.activate([0,1])
print '1,0->', net.activate([1,0])
print '1,1->', net.activate([1,1])

print '0.2,0.2->', net.activate([0.2,0.2])
print '0.2,0.8->', net.activate([0.2,0.8])
print '0.8,0.2->', net.activate([0.8,0.2])
print '0.8,0.8->', net.activate([0.8,0.8])
