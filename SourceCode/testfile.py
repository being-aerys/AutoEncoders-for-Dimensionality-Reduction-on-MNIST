import functools
import operator

joint_state_list_of_lists = [[1,1,1],[2,2],[3,3,3,3]]
print("reduced list ",functools.reduce(operator.iconcat, joint_state_list_of_lists, []))
