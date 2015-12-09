import csv

from pybrain.tools.shortcuts     import buildNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure           import *

# download from https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data
fname = "car.data"

# in_layer      = LinearLayer(6, name='in')
# hidden_layer1 = SigmoidLayer(8, name='hid')
# out_layer     = LinearLayer(1, name='out')
# bias_unit     = BiasUnit(name='bias')

# net = FeedForwardNetwork()

# net.addInputModule(in_layer)
# net.addModule(hidden_layer1)
# net.addModule(bias_unit)
# net.addOutputModule(out_layer)

# in_to_hidden1   = FullConnection(in_layer, hidden_layer1)
# hidden1_to_out  = FullConnection(hidden_layer1, out_layer)
# bias_to_hidden1 = FullConnection(bias_unit, hidden_layer1)
# bias_to_out     = FullConnection(bias_unit, out_layer)

# net.addConnection(in_to_hidden1)
# net.addConnection(hidden1_to_out)
# net.addConnection(bias_to_hidden1)
# net.addConnection(bias_to_out)

# net.sortModules()

net = buildNetwork(6, 8, 1, bias=True, hiddenclass=SigmoidLayer)

ds = SupervisedDataSet(6, 1)

trainer = BackpropTrainer(net, learningrate=0.01, momentum=0.99)

# _convertToOneOfMany   # exercise: what is it?

def rev_map(dictionary):
    return dict((reversed(item) for item in dictionary.items()))

price    = { 'vhigh': 3, 'high': 2, 'med': 1, 'low': 0 };
doors    = { '2': 3, '3': 2, '4': 1, '5more': 0 };
persons  = { '2': 2, '4': 1, 'more': 0 };
lug_boot = { 'small': 2, 'med': 1, 'big': 0 };
safety   = { 'low': 2, 'med': 1, 'high': 0 };
result   = { 'vgood': 3, 'good': 2, 'acc': 1, 'unacc': 0 };

price_r    = rev_map(price)
doors_r    = rev_map(doors)
persons_r  = rev_map(persons)
lug_boot_r = rev_map(lug_boot)
safety_r   = rev_map(safety)
result_r   = rev_map(result)

# buying       vhigh, high, med, low
# maint        vhigh, high, med, low
# doors        2, 3, 4, 5more
# persons      2, 4, more
# lug_boot     small, med, big
# safety       low, med, high

with open(fname, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        sample = (price[row[0]], price[row[1]], doors[row[2]], persons[row[3]], lug_boot[row[4]], safety[row[5]])
        ds.addSample(sample, result[row[6]])

tst_ds, trn_ds = ds.splitWithProportion(0.2)

# print "train data"
# for inpt, target in trn_ds:
#     print inpt, target

# print "test data"
# for inpt, target in tst_ds:
#     print inpt, target

# More information about trainers: http://pybrain.org/docs/api/supervised/trainers.html

print "Training started"

trainer.trainOnDataset(trn_ds, 10)

# trainer.trainUntilConvergence(trn_ds, maxEpochs=100, verbose=True, continueEpochs=10, validationProportion=0.25)

# trainer.testOnData(tst_ds, verbose=True)


# exercise: how to get network error value for a given epoch?

# from: http://stackoverflow.com/questions/8150772/pybrain-how-to-print-a-network-nodes-and-weights
def display_net(net):
    for mod in net.modules:
        print("Module:", mod.name)
        if mod.paramdim > 0:
            print("--parameters:", mod.params)
        for conn in net.connections[mod]:
            print("-connection to", conn.outmod.name)
            if conn.paramdim > 0:
                 print("- parameters", conn.params)
        if hasattr(net, "recurrentConns"):
            print("Recurrent connections")
            for conn in net.recurrentConns:
                print("-", conn.inmod.name, " to", conn.outmod.name)
                if conn.paramdim > 0:
                    print("- parameters", conn.params)

# display_net(net)

verbose = True

def display_result(idx):
    row = ds['input'][idx]
    exp = ds['target'][idx][0]
    res = net.activate(row)
    exp_str = result_r[exp]
    res_str = result_r[min(max(round(res), 0), 3)]
    matched = ""
    if res_str != exp_str:
        matched = "not matched"

    if verbose:
        sample = (price_r[row[0]], price_r[row[1]], doors_r[row[2]], persons_r[row[3]], lug_boot_r[row[4]], safety_r[row[5]])
        print "result:   %7s %f" % (res_str, res), "------- row ", idx, row, " values: %s, %s, %s, %s, %s, %s" % sample
        print "expected: %7s %d" % (exp_str, exp), matched

    if matched == "":
        return 0
    else:
        return 1


allnum = 1728
missed = 0
for i in range(0, allnum):
    missed += display_result(i)

print "Not matched %s (%.2f%%)" % (missed, 100.0 * missed / allnum)
