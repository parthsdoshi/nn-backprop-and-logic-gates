####################### BLOG CHECK #######################
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# import torch
# from neural_network import NeuralNetwork

# nn = NeuralNetwork([2, 2, 2])
# nn.theta = [
#     torch.tensor([
#         [.35, .15, .20],
#         [.35, .25, .30]
#     ]).t(),
#     torch.tensor([
#         [.60, .40, .45],
#         [.60, .50, .55]
#     ]).t(),
# ]
# x = torch.tensor([[.05, .1]])
# y = nn.forward(x)
# print(y)
# Y = torch.tensor([[.01, .99]])
# e = nn.mse(x, Y)
# print(e * .5)

# nn.backward(Y)
# print(nn.dE_dTheta)
# nn.updateParams(.5)
# print(nn.theta)
# print(nn.delta)
##########################################################

from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR

# wrong = 0
# correct = 0
# for i in range(100):
#     xor_gate = XOR()
#     xor_gate.train()
#     mse = xor_gate._mse()
#     if mse >= .01:
#         wrong += 1
#     else:
#         correct += 1

# print(wrong)
# print(correct)
# print(correct / (wrong + correct))

# exit()

and_gate = AND()
and_gate.train()

or_gate = OR()
or_gate.train()

not_gate = NOT()
not_gate.train()

xor_gate = XOR()
xor_gate.train()

print("")

print("AND GATE")
print(and_gate(False, False))
print(and_gate(False, True))
print(and_gate(True, False))
print(and_gate(True, True))
print("AND GATE WEIGHTS")
print(and_gate.model.theta)
print("")

print("OR GATE")
print(or_gate(False, False))
print(or_gate(False, True))
print(or_gate(True, False))
print(or_gate(True, True))
print("OR GATE WEIGHTS")
print(or_gate.model.theta)
print("")

print("NOT GATE")
print(not_gate(False))
print(not_gate(True))
print("NOT GATE WEIGHTS")
print(not_gate.model.theta)
print("")

print("XOR GATE")
print(xor_gate(False, False))
print(xor_gate(False, True))
print(xor_gate(True, False))
print(xor_gate(True, True))
print("XOR GATE WEIGHTS")
print(xor_gate.model.theta)
print("")