import numpy as np
import pulp


prob = pulp.LpProblem("GHP_toy_problem", pulp.LpMinimize)

#define slots:
t = [[1600, 1601], [1602, 1603], [1604, 1605]]

#define capacity of the slot
b = [1, 2, 1]

ETA = [1600, 1600, 1602, 1603]
r = [1, 3, 1, 5]

def cost (r: int, t: list[int], ETA: int):
    costval = r*(t[0]-ETA)
    return costval


# Definir costos para cada variable (ejemplo - ajusta según tus necesidades)
costs = {
    'x1': cost(r[0], t[0], ETA[0]), 'x2': cost(r[0], t[1], ETA[0]), 'x3': cost(r[0], t[2], ETA[0]),
    'x4': cost(r[1], t[0], ETA[1]), 'x5': cost(r[1], t[1], ETA[1]), 'x6': cost(r[1], t[2], ETA[1]),
    'x7': cost(r[2], t[0], ETA[2]), 'x8': cost(r[2], t[1], ETA[2]), 'x9': cost(r[2], t[2], ETA[2]),
    'x10': cost(r[3], t[0], ETA[3]), 'x11': cost(r[3], t[1], ETA[3]), 'x12': cost(r[3], t[2], ETA[3])
}

# Corregir nombres de variables (tenías nombres duplicados)
x1 = pulp.LpVariable("x1", cat='binary')
x2 = pulp.LpVariable("x2", cat='binary')
x3 = pulp.LpVariable("x3", cat='binary')
x4 = pulp.LpVariable("x4", cat='binary')
x5 = pulp.LpVariable("x5", cat='binary')
x6 = pulp.LpVariable("x6", cat='binary')
x7 = pulp.LpVariable("x7", cat='binary')
x8 = pulp.LpVariable("x8", cat='binary')
x9 = pulp.LpVariable("x9", cat='binary')
x10 = pulp.LpVariable("x10", cat='binary')
x11 = pulp.LpVariable("x11", cat='binary')
x12 = pulp.LpVariable("x12", cat='binary')

# Función objetivo: Minimizar el costo total
prob += (
    costs['x1'] * x1 + costs['x2'] * x2 + costs['x3'] * x3 +
    costs['x4'] * x4 + costs['x5'] * x5 + costs['x6'] * x6 +
    costs['x7'] * x7 + costs['x8'] * x8 + costs['x9'] * x9 +
    costs['x10'] * x10 + costs['x11'] * x11 + costs['x12'] * x12,
    "Total_Cost"
)

# Constraints
prob += x1 + x4 + x7 + x10 <= 1, "No surpass the slot 1 capacity"
prob += x2 + x5 + x8 + x11 <= 1, "No surpass the slot 2 capacity"
prob += x3 + x6 + x9 + x12 <= 1, "No surpass the slot 3 capacity"

prob += x1 + x2 + x3 == 1, "Fligth arrives once at one slot 1"
prob += x4 + x5 + x6 == 1, "Fligth arrives once at one slot 2"
prob += x7 + x8 + x9 == 1, "Fligth arrives once at one slot 3"

prob += x7 == 0, "SLOT1 < ETA"
prob += x10 == 0, "SLOT< ETA "

prob += 0 <= x1 <= 1, "x1 Binary variable"
prob += 0 <= x2 <= 1, "x2 Binary variable"
prob += 0 <= x3 <= 1, "x3 Binary variable"
prob += 0 <= x4 <= 1, "x4 Binary variable"
prob += 0 <= x5 <= 1, "x5 Binary variable"
prob += 0 <= x6 <= 1, "x6 Binary variable"
prob += 0 <= x7 <= 1, "x7 Binary variable"
prob += 0 <= x8 <= 1, "x8 Binary variable"
prob += 0 <= x9 <= 1, "x9 Binary variable"
prob += 0 <= x10 <= 1, "x10 Binary variable"
prob += 0 <= x11 <= 1, "x11 Binary variable"
prob += 0 <= x12 <= 1, "x12 Binary variable"




prob.solve()

for var in prob.variables():
    print(f"{var.name}: {var.varValue}")

print(f"Minimum Cost: {pulp.value(prob.objective)}")





