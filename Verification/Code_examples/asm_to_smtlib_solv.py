#!/usr/bin/env python

# EVM disassembler
from manticore.platforms.evm import *
from manticore.core.smtlib import *
from manticore.core.smtlib.visitors import *
from manticore.utils import log

from manticore.core.smtlib.solver import Z3Solver

# log.set_verbosity(9)
config.out_of_gas = 1

constraints = ConstraintSet()

# code = EVMAsm.assemble(
#     """
#     INVALID
#     """
#  )

cs = ConstraintSet()
# new_array, new_bool
# x = int()
# y = int()
# z = int()

x = cs.new_bitvec(32, name="x")
y = cs.new_bitvec(32, name="y")
# mm = cs.new_array()
sender = cs.new_bitvec(32, name="sender")

cs.add(x > 10)
cs.add(y > 10)
c_val = cs.new_bitvec(32, name="c_val")
cs.add( c_val == (x + y))
cs.add( c_val < 20)

# # cs.add(y < 2)
# # # cs.add(y < 2, (x+y) == 10,  x > 5)
# # cs.add((x+y) < 10)
# # cs.add(x < 5)
# # cs.add(z == True)
# # cs.add(sender == addr)
# # # result = x, y

print("CONSTRAINTS:")
print(cs)

# # Well... x is symbolic
smtsolver = Z3Solver.instance()
print("The path feasibility: {}, checked by {} solver!".format(smtsolver.check(constraints), smtsolver.sname))
# # # if we ask for a possible solution (an x that complies with the constraints)
x, y = smtsolver.get_value(cs, x, y)
print(f"x={x}")
print(f"y={y}")

smtsolver = YicesSolver()
print("The path feasibility: {}, checked by {} solver!".format(smtsolver.check(constraints), smtsolver.sname))
# # # if we ask for a possible solution (an x that complies with the constraints)
x, y = smtsolver.get_value(cs, x, y)
print(f"x={x}")
print(f"y={y}")

smtsolver = CVC4Solver()
print("The path feasibility: {}, checked by {} solver!".format(smtsolver.check(constraints), smtsolver.sname))
# # # if we ask for a possible solution (an x that complies with the constraints)
x, y = smtsolver.get_value(cs, x, y)
print(f"x={x}")
print(f"y={y}")