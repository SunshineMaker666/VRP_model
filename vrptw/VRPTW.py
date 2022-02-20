import numpy as np
import pandas as pd
import math
from pulp import *
from scipy.spatial.distance import pdist, squareform


#获取测试数据
data = np.genfromtxt('VRPTW.txt')

#坐标提取
X = data[:,1]#X坐标
Y = data[:,2]#Y坐标
q = data[:,3]#需求
E = data[:,4]#时间窗E
L = data[:,5]#时间窗L
S = data[:,6]#服务时间

M = 1000000000#超大数

axis=np.vstack([X,Y]).T
a=pdist(axis)
costs=squareform(a)

I = [i for i in range(1,13)]#客户编号
I0 = [0] + I
In = I0 + [13]
K = [i for i in range(0,3)]#车辆数

Q = 70#车容量

#创建问题实例，求最小极值
prob = LpProblem("The CVRP Problem", LpMinimize)


#构建Lp变量字典，变量名以Ingr开头，如Ingr_CHICKEN，下界是0
car_vars = [[pulp.LpVariable('y%d_%d'%(k,i),cat = LpBinary) for i in In] for k in K]
route_vars = [[[pulp.LpVariable('x%d_%d_%d'%(k,i,j),cat = LpBinary) for j in In] for i in I0] for k in K]
u_vars = [pulp.LpVariable('u%d'%i, cat = LpContinuous) for i in I]
t_vars = [[pulp.LpVariable('t%d_%d'%(k,i),lowBound = 0,cat = LpContinuous) for i in In] for k in K]

#添加目标方程
prob += lpSum([costs[i,j]*route_vars[k][i][j] for k in K for i in I0 for j in In])

#prob += lpSum([t_vars[k][13] for k in K])

#添加约束条件
#每个点都要配一辆车
for i in I:
    prob += lpSum(car_vars[k][i] for k in K) == 1 

#流量平衡
for k in K:
    for j in I:
        prob += lpSum(route_vars[k][i][j] for i in I0 if i!=j) == lpSum(route_vars[k][j][i] for i in In if i!=j)
        
#车辆出发在depot
for k in K:
    prob += lpSum(route_vars[k][0][j] for j in In) == 1

#车辆返回在depot
for k in K:
    prob += lpSum(route_vars[k][i][0] for i in I0) == 1
    

#车辆载重限制
for k in K:
    prob += lpSum(q[i]*car_vars[k][i] for i in I)<=Q

#变量关系
for k in K:
    for j in I:
        prob += lpSum(route_vars[k][i][j] for i in I0 if i!=j) == car_vars[k][j]

#变量约束
for k in K:
    for i in I0:
        for j in In:
            if i==j:
                prob += route_vars[k][i][j] == 0

#消除子回路
for k in K:
    for i in I0:
        for j in I0:
            prob += route_vars[k][i][j] + route_vars[k][j][i] <=1


for k in K:
    for i in I:
        for j in I:
            if i!=j: 
                prob += u_vars[i-1] - u_vars[j-1] + len(I)*route_vars[k][i][j] <= len(I)-1

#时间约束
for k in K:
    for i in I0:
        for j in I:
            if i!=j:
                prob += t_vars[k][i] + S[i] + costs[i][j] - M*(1-route_vars[k][i][j])<=t_vars[k][j]
                         
for k in K:
    for i in I:
        prob += t_vars[k][i] <= L[i]
        #prob += t_vars[k][i] >= E[i]
        prob += t_vars[k][i] <= car_vars[k][i]*M
        
'''
#车辆必须回到depot
for k in K:
    prob+= car_vars[k][13] == 1
'''
#求解
prob.solve(PULP_CBC_CMD(fracGap = 0.00001, maxSeconds = 100, threads = None))
#查看解的状态
print("Status:", LpStatus[prob.status])
#查看解
for v in prob.variables():
    if v.varValue>0:
        print(v.name, "=", v.varValue)

print('Obj: %g' % prob.objective.value())
