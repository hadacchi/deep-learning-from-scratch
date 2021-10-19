import numpy
import timeit

w1 = 0.5
w2 = 0.5
w = numpy.array([w1, w2])
theta = 0.7

# perceptron
def NODE(w1, w2, theta):
    return lambda x1, x2: 1 if x1*w1 + x2*w2 > theta else 0

AND = NODE(0.5, 0.5, 0.7)
OR = NODE(0.5, 0.5, 0.1)
NAND = NODE(-0.5, -0.5, -0.7)

#def AND(x1, x2):
#    return 1 if x1*w1 + x2*w2 > theta else 0
#
#def AND_np(x1, x2):
#    x = numpy.array([x1,x2])
#    return 1 if numpy.sum(x*w) > theta else 0
#
#def NAND(x1, x2):
#    return 0 if x1*w1 + x2*w2 > theta else 1
#
#def OR(x1, x2):
#    return 1 if x1*w1 + x2*w2 > 0 else 0
#
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

#loop = 10000
#
#result = timeit.timeit('AND(0,1)', globals=globals(), number=loop)
#print(result/loop)
#
#result = timeit.timeit('AND_np(0,1)', globals=globals(), number=loop)
#print(result/loop)

print('AND')
print(f'0 0 {AND(0,0)}')
print(f'0 1 {AND(0,1)}')
print(f'1 0 {AND(1,0)}')
print(f'1 1 {AND(1,1)}')

print('NAND')
print(f'0 0 {NAND(0,0)}')
print(f'0 1 {NAND(0,1)}')
print(f'1 0 {NAND(1,0)}')
print(f'1 1 {NAND(1,1)}')

print('OR')
print(f'0 0 {OR(0,0)}')
print(f'0 1 {OR(0,1)}')
print(f'1 0 {OR(1,0)}')
print(f'1 1 {OR(1,1)}')

print('XOR')
print(f'0 0 {XOR(0,0)}')
print(f'0 1 {XOR(0,1)}')
print(f'1 0 {XOR(1,0)}')
print(f'1 1 {XOR(1,1)}')

