import gamspy as gp
import numpy as np
import pandas as pd


from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense, Options

# =============================================================
# Questions a, b1, b2 — Deterministic Model, Base problem
# =============================================================

m = Container()

# Define the indices: i the products, j the factories, k the candidate distribution centers, l the customers
i = Set(m, name='i', records=["PLS 502", "PLS 502L", "PLS 602", "PLS 602S", "PLS 602G", "PLS 602SF", "PLS 702", "PLS 702S", "PLS 702SF", "PLS 103NF", "M1", "JK1"], description="Product")
j = Set(m, name='j', records=["Nea Santa", "Komotini"], description="Factories")
k = Set(m, name='k', records=["Wurzburg", "Hanover", "Magdeburg", "Erfurt", "Ingolstadt", "Bielefeld", "Cologne", "Munich"], description="Distribution Centers")
l = Set(m, name='l', records=["Hamburg", "Berlin", "Cologne", "Frankfurt", "Munich", "Dusseldorf", "Dresden", "Nuremberg", "Bonn", "Leipzig"], description="Customer Zone / Agent")

# Define the parameters
# Production Cost (c)
c = Parameter(
    container=m,
    name="c",
    domain=[i, j],
    description="Unit production cost of product i at factory j (EUR/unit)"
)
# Installation Cost b (building)
b = Parameter(
    container=m,
    name="b",
    domain=[k],
    description="Installation cost at distribution center k (EUR)"
)
# Variable Handling Cost
h = Parameter(
    container=m,
    name="h",
    domain=[k],
    description="Handling cost at distribution center k (EUR/unit)"
)
# Unit storage volume at the DC (Volume)
v = Parameter(
    container=m,
    name="v",
    domain=[i, k],
    description="Storage capacity for one unit of product i at distribution center k (m2/unit)"
)
# Transport cost from factory to DC (Cost to Distribution, cd)
cd = Parameter(
    container=m,
    name="cd",
    domain=[j, k],
    description="Unit transport cost of products from factory j to distribution center k (EUR/unit)"
)
# Transport cost from DC to customer (Cost to Customer, cc)
cc = Parameter(
    container=m,
    name="cc",
    domain=[k, l],
    description="Unit transport cost of products from distribution center k to customer/agent zone l (EUR/unit)"
)
# Demand
d = Parameter(
    container=m,
    name="d",
    domain=[i, l],
    description="Demand for product i from customer/agent zone l (units)"
)

# Define the decision variables
x = Variable(
    container=m,
    name="x",
    domain=[i, j, k],
    type="Positive",
    description="Production quantity of product i at factory j transported to DC k",
)
w = Variable(
    container=m,
    name="w",
    domain=[i, k, l],
    type="Positive",
    description="Quantity of product i leaving DC k for customer l",
)

y = Variable(
    container=m,
    name="y",
    domain=[k],
    type="binary",
    description="Decision variable on whether to open DC k. "
    "If opened yk = 1. "
    "If not opened yk = 0",
)

z = Variable(
    m,
    name="objective"
)


# Pass the values of the tables

# Production Cost c[i,j]
df_prod = pd.DataFrame({
    "Product": ["PLS 502", "PLS 502L", "PLS 602", "PLS 602S", "PLS 602G", "PLS 602SF", "PLS 702", "PLS 702S", "PLS 702SF", "PLS 103NF", "M1", "JK1"],
    "Nea Santa": [84.66, 92.60, 83.88, 94.18, 91.85, 110.47, 110.17, 119.79, 114.80, 128.71, 15.77, 57.57],
    "Komotini": [780.45, 250.66, 320.45, 155.54, 135.47, 221.66, 248.42, 287.71, 187.38, 165.78, 5.62, 11.55]
})
df_prod_long = df_prod.melt(id_vars="Product", var_name="Factories", value_name="Production Cost")
c.setRecords(df_prod_long)

# Unit storage volume at the distribution center
df_vol = pd.DataFrame({
    "Product": ["PLS 502", "PLS 502L", "PLS 602", "PLS 602S", "PLS 602G", "PLS 602SF", "PLS 702", "PLS 702S", "PLS 702SF", "PLS 103NF", "M1", "JK1"],
    "Wurzburg":   [0.02, 0.03, 0.01, 0.01, 0.03, 0.02, 0.01, 0.05, 0.04, 0.02, 0.03, 0.02],
    "Hanover":    [0.01, 0.02, 0.01, 0.06, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.04, 0.04],
    "Magdeburg":  [0.04, 0.04, 0.04, 0.06, 0.06, 0.02, 0.01, 0.02, 0.02, 0.03, 0.03, 0.02],
    "Erfurt":     [0.02, 0.02, 0.02, 0.04, 0.07, 0.02, 0.02, 0.04, 0.02, 0.02, 0.02, 0.02],
    "Ingolstadt": [0.02, 0.02, 0.02, 0.04, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.02],
    "Bielefeld":  [0.02, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02, 0.03, 0.03, 0.01, 0.01],
    "Cologne":    [0.04, 0.02, 0.05, 0.07, 0.02, 0.07, 0.07, 0.02, 0.01, 0.02, 0.06, 0.02],
    "Munich":     [0.03, 0.02, 0.03, 0.02, 0.09, 0.02, 0.04, 0.02, 0.09, 0.02, 0.04, 0.07]
})

df_vol_long = df_vol.melt(id_vars="Product", var_name="Distribution Centers", value_name="Volume")
v.setRecords(df_vol_long)

# Installation Cost
df_b = pd.DataFrame({
    "Distribution Centers": ["Wurzburg", "Hanover", "Magdeburg", "Erfurt", "Ingolstadt", "Bielefeld", "Cologne", "Munich"],
    "Installation": [90740, 102570, 80540, 96605, 102570, 96605, 96605, 108435]
})
b.setRecords(df_b)

# Unit Handling Cost
df_h = pd.DataFrame({
    "Distribution Centers": ["Wurzburg", "Hanover", "Magdeburg", "Erfurt", "Ingolstadt", "Bielefeld", "Cologne", "Munich"],
    "Handling": [0.516, 0.716, 0.516, 0.614, 0.716, 0.614, 0.614, 0.818]
})
h.setRecords(df_h)

# Customer demand
df_d = pd.DataFrame({
    "Product": ["PLS 502", "PLS 502L", "PLS 602", "PLS 602S", "PLS 602G", "PLS 602SF", "PLS 702", "PLS 702S", "PLS 702SF", "PLS 103NF", "M1", "JK1"],
    "Hamburg":    [95, 140, 115, 140, 65, 85, 55, 60, 35, 165, 745, 7165],
    "Berlin":     [365, 290, 65, 275, 90, 150, 70, 135, 90, 340, 3276, 5860],
    "Cologne":    [1240, 1110, 845, 440, 210, 310, 565, 205, 105, 235, 4765, 7160],
    "Frankfurt":  [370, 255, 305, 170, 115, 115, 200, 140, 85, 470, 3425, 10420],
    "Munich":     [900, 485, 320, 290, 145, 120, 325, 190, 130, 105, 2680, 7280],
    "Dusseldorf": [265, 65, 140, 172, 180, 220, 50, 65, 38, 175, 135, 135],
    "Dresden":    [30, 60, 90, 140, 125, 185, 190, 110, 95, 78, 145, 163],
    "Nuremberg":  [140, 125, 195, 135, 178, 163, 210, 720, 358, 471, 105, 297],
    "Bonn":       [180, 1600, 350, 140, 187, 373, 506, 105, 197, 120, 172, 263],
    "Leipzig":    [175, 105, 284, 180, 180, 295, 170, 348, 130, 220, 270, 230]
})
df_d_long = df_d.melt(id_vars="Product", var_name="Customer Zone / Agent", value_name="Demand")
d.setRecords(df_d_long)

# Unit transport cost from factory to DC
df_cd = pd.DataFrame({
    "Factories": ["Nea Santa", "Komotini"],
    "Wurzburg":   [13.50, 14.96],
    "Hanover":    [14.08, 15.55],
    "Magdeburg":  [14.38, 15.84],
    "Erfurt":     [13.20, 14.67],
    "Ingolstadt": [16.42, 15.86],
    "Bielefeld":  [13.79, 15.26],
    "Cologne":    [14.96, 16.43],
    "Munich":     [13.79, 15.26]
})
df_cd_long = df_cd.melt(id_vars="Factories", var_name="Distribution Centers", value_name="Transport Cost Factory - DC")
cd.setRecords(df_cd_long)

# Unit transport cost from DC to Customers
df_cc = pd.DataFrame({
    "Distribution Centers": ["Wurzburg", "Hanover", "Magdeburg", "Erfurt", "Ingolstadt", "Bielefeld", "Cologne", "Munich"],
    "Hamburg":    [3.48, 1.02, 1.18, 2.03, 3.38, 1.84, 0.68, 0.68],
    "Berlin":     [3.96, 1.92, 1.01, 1.52, 2.36, 2.65, 0.68, 0.68],
    "Cologne":    [2.63, 2.16, 2.84, 2.57, 3.98, 1.49, 0, 3.71],
    "Frankfurt":  [0.77, 2.38, 1.18, 1.69, 2.19, 1.14, 0.68, 0.68],
    "Munich":     [1.82, 5.06, 4.19, 2.70, 2.54, 5.06, 3.71, 0],
    "Dusseldorf": [1.63, 1.01, 0.78, 2.32, 0.77, 0.84, 0.61, 2.10],
    "Dresden":    [2.38, 4.63, 2.01, 0.93, 1.78, 1.32, 0.91, 1.94,],
    "Nuremberg":  [3.42, 3.07, 1.45, 1.92, 1.38, 2.27, 0, 3.51],
    "Bonn":       [4.07, 1.94, 3.33, 0, 2.58, 1.44, 3.72, 1.69],
    "Leipzig":    [3.17, 2.12, 2.27, 1.18, 3.51, 2.02, 2.33, 1.82]
})
df_cc_long = df_cc.melt(id_vars="Distribution Centers", var_name="Customer Zone / Agent", value_name="Transport Cost DC - Customers")
cc.setRecords(df_cc_long)


# Define the constraints of the problem
# Definition of the Equations
demand_con = Equation(m, name="demand_con", domain=[i, l], description="Demand satisfaction")
balance_con = Equation(m, name="balance_con", domain=[i, k], description="Flow balance at the DC")
one_dc_con = Equation(m, name="one_dc_con", description="Select exactly one DC")
capacity_dc_con = Equation(m, name="capacity_dc_con", domain=[k], description="DC capacity in m2")
transport_limit_con = Equation(m, name="transport_limit_con", domain=[j, k], description="Transport limit 90,000 from factory to DC")
transport_limit_con2 = Equation(m, name="transport_limit_con2", domain=[k, l], description="Transport limit 90,000 from DC to customers")

# The demand of each customer l for product i must be satisfied by the DC
demand_con[i, l] = gp.Sum(k, w[i, k, l]) == d[i, l]

# Whatever enters DC k from the factories j must leave towards the customers l
balance_con[i, k] = gp.Sum(j, x[i, j, k]) == gp.Sum(l, w[i, k, l])


# Exactly one DC must be opened
one_dc_con[...] = gp.Sum(k, y[k]) == 1


# Total volume (m2) at DC k <= 50,000
capacity_dc_con[k] = gp.Sum((i, l), v[i, k] * w[i, k, l]) <= 50000 * y[k]


# Total flow from factory j to DC k <= 90,000
transport_limit_con[j, k] = gp.Sum(i, x[i, j, k]) <= 90000 * y[k]


# Total flow from DC k to customer l <= 90,000
transport_limit_con2[k, l] = gp.Sum(i, w[i, k, l]) <= 90000 * y[k]


# Definition of the objective function
obj_equation = Equation(m, "obj_equation")

obj_equation[:] = (
    z ==
    Sum((i, j, k), c[i, j] * x[i, j, k]) +
    Sum(k, b[k] * y[k]) +
    Sum((i, k, l), h[k] * w[i, k, l]) +
    Sum((i, j, k),  cd[j, k] * x[i, j, k]) +
    Sum((i, k, l), cc[k, l] * w[i, k, l])
)



# Definition of the solver model
network_model = Model(
    m,
    name="network_model",
    equations=m.getEquations(),
    problem="MIP",
    sense=Sense.MIN,
    objective=z
)

# Solve the problem
network_model.solve()

# Display the Optimal Solution results
print("\n========== Optimal Solution ==========\n")

optimal_cost = z.records["level"][0]

print(f"Minimum Total Cost = {optimal_cost:.2f} EUR")


selected_dc = y.records[y.records["level"] > 0.5]

print("\nSelected Distribution Center:")
print(selected_dc)

selected_dc_name = selected_dc.iloc[0]["k"]

print(f"\nSelected Distribution Center: {selected_dc_name}")


print("\n========== Factory -> Flows of the selected DC  ==========\n")

x_selected = x.records[
    (x.records["k"] == selected_dc_name) &
    (x.records["level"] > 0)
]

print(x_selected)

# Filter for the DC and for level > 0
w_selected = w.records[
    (w.records["k"] == selected_dc_name) &
    (w.records["level"] > 1e-6)  # Use a small tolerance instead of 0
].copy()

# Create Pivot Table
pivot_results = w_selected.pivot_table(
    index="l",
    columns="i",
    values=["level", "upper", "marginal"],
    aggfunc="sum",
    fill_value=0
)

x_df = x.records[x.records["level"] > 0].copy()
w_df = w.records[w.records["level"] > 0].copy()
y_df = y.records.copy()


print("c columns:", c.records.columns.tolist())
print("cd columns:", cd.records.columns.tolist())
print("h columns:", h.records.columns.tolist())
print("cc columns:", cc.records.columns.tolist())
print("b columns:", b.records.columns.tolist())

x_df = x.records[x.records["level"] > 0][["i", "j", "k", "level"]].copy()
w_df = w.records[w.records["level"] > 0][["i", "k", "l", "level"]].copy()


c_vals = c.records.copy()
c_vals.columns = ["i", "j", "prod_cost"]
production_cost = (
    x_df.merge(c_vals, on=["i", "j"])
    .assign(cost=lambda df: df["level"] * df["prod_cost"])
    ["cost"].sum()
)

# Transport Cost Factory -> DC
cd_vals = cd.records.copy()
cd_vals.columns = ["j", "k", "trans_jk_cost"]
transport_jk_cost = (
    x_df.merge(cd_vals, on=["j", "k"])
    .assign(cost=lambda df: df["level"] * df["trans_jk_cost"])
    ["cost"].sum()
)

# DC Handling Cost
h_vals = h.records.copy()
h_vals.columns = ["k", "handling_cost"]
handling_cost = (
    w_df.merge(h_vals, on="k")
    .assign(cost=lambda df: df["level"] * df["handling_cost"])
    ["cost"].sum()
)

# Transport Cost DC -> Customers
cc_vals = cc.records.copy()
cc_vals.columns = ["k", "l", "trans_kl_cost"]
transport_kl_cost = (
    w_df.merge(cc_vals, on=["k", "l"])
    .assign(cost=lambda df: df["level"] * df["trans_kl_cost"])
    ["cost"].sum()
)

# Fixed Installation Cost
b_vals = b.records.copy()
b_vals.columns = ["k", "fixed_cost"]
y_vals = y.records[["k", "level"]].copy()
fixed_cost = (
    y_vals.merge(b_vals, on="k")
    .assign(cost=lambda df: df["level"] * df["fixed_cost"])
    ["cost"].sum()
)

total_check = production_cost + transport_jk_cost + handling_cost + transport_kl_cost + fixed_cost

print("\n========== Cost Breakdown ==========")
print(f"  Production:                  {production_cost:>15,.2f} EUR")
print(f"  Transport Factory -> DC:     {transport_jk_cost:>15,.2f} EUR")
print(f"  DC Handling:                 {handling_cost:>15,.2f} EUR")
print(f"  Transport DC -> Customers:   {transport_kl_cost:>15,.2f} EUR")
print(f"  DC Installation (fixed):     {fixed_cost:>15,.2f} EUR")
print(f"  {'-'*38}")
print(f"  Total (manual check):         {total_check:>15,.2f} EUR")
print(f"  Total (solver):               {z.records['level'][0]:>15,.2f} EUR")

print("\n" + "="*70)
print("  STOCHASTIC MODEL - Finding Optimal DC (k) across all scenarios")
print("="*70)

# NEW INDEX
s = Set(m, name="s",
        records=["lower", "low", "basic", "high", "higher"],
        description="Demand Scenarios")

# NEW PARAMETERS
prob = Parameter(m, name="prob", domain=[s])
prob.setRecords([
    ("lower",  0.05),
    ("low",    0.15),
    ("basic",  0.60),
    ("high",   0.15),
    ("higher", 0.05),
])

scenario_factors = {
    "lower": 0.75,
    "low":   0.90,
    "basic": 1.00,
    "high":  1.10,
    "higher": 1.25,
}

d_s = Parameter(m, name="d_s", domain=[i, l, s],
                description="Scenario demand for product i at customer l")

demand_base = d.records.copy()
demand_base.columns = ["product", "customer", "base_demand"]
rows = []
for scen, fac in scenario_factors.items():
    tmp = demand_base.copy()
    tmp["scenario"] = scen
    tmp["scenario_demand"] = np.ceil(tmp["base_demand"] * fac).astype(int)
    rows.append(tmp)
scen_df = pd.concat(rows, ignore_index=True)[["product", "customer", "scenario", "scenario_demand"]]
scen_df.columns = ["i", "l", "s", "value"]
d_s.setRecords(scen_df)

# NEW VARIABLES
x_s = Variable(m, name="x_s", domain=[i, j, k, s], type="Positive",
               description="Production of i at factory j sent to DC k under scenario s")
w_s = Variable(m, name="w_s", domain=[i, k, l, s], type="Positive",
               description="Flow of i from DC k to customer l under scenario s")
z_s = Variable(m, name="z_s", domain=[s],
               description="Total cost under scenario s")
z_expected = Variable(m, name="z_expected",
                      description="Expected total stochastic cost")

# CONSTRAINTS
demand_s   = Equation(m, "demand_s",   domain=[i, l, s])
balance_s  = Equation(m, "balance_s",  domain=[i, k, s])
one_dc_s   = Equation(m, "one_dc_s")
cap_s      = Equation(m, "cap_s",      domain=[k, s])
tlim_jk_s  = Equation(m, "tlim_jk_s", domain=[j, k, s])
tlim_kl_s  = Equation(m, "tlim_kl_s", domain=[k, l, s])
cost_s_eq  = Equation(m, "cost_s_eq",  domain=[s])
exp_eq     = Equation(m, "exp_eq")

demand_s[i, l, s]      = Sum(k, w_s[i, k, l, s]) == d_s[i, l, s]
balance_s[i, k, s]     = Sum(j, x_s[i, j, k, s]) == Sum(l, w_s[i, k, l, s])
one_dc_s[...]          = Sum(k, y[k]) == 1                   # reuse y[k]
cap_s[k, s]            = Sum((i, l), v[i, k] * w_s[i, k, l, s]) <= 50000 * y[k]
tlim_jk_s[j, k, s]     = Sum(i, x_s[i, j, k, s]) <= 90000 * y[k]
tlim_kl_s[k, l, s]     = Sum(i, w_s[i, k, l, s]) <= 90000 * y[k]

cost_s_eq[s] = (
    z_s[s] ==
    Sum(k, b[k] * y[k]) +
    Sum((i, j, k), c[i, j]  * x_s[i, j, k, s]) +
    Sum((i, k, l), h[k]     * w_s[i, k, l, s]) +
    Sum((i, j, k), cd[j, k] * x_s[i, j, k, s]) +
    Sum((i, k, l), cc[k, l] * w_s[i, k, l, s])
)
exp_eq[...] = z_expected == Sum(s, prob[s] * z_s[s])


stoch_model = Model(
    m,
    name="stoch_model",
    equations=[demand_s, balance_s, one_dc_s, cap_s,
               tlim_jk_s, tlim_kl_s, cost_s_eq, exp_eq],
    problem="MIP",
    sense=Sense.MIN,
    objective=z_expected,
)


dc_candidates = ["Wurzburg", "Hanover", "Magdeburg", "Erfurt",
                 "Ingolstadt", "Bielefeld", "Cologne",    "Munich"]

results_per_dc = []

print("\nSolving stochastic model for each DC candidate ...\n")
print(f"{'DC Candidate':<20}  {'Expected Cost (EUR)':>20}")
print("-" * 44)

for dc in dc_candidates:

    y_fix = pd.DataFrame({
        "k":     dc_candidates,
        "level": [1.0 if dc_k == dc else 0.0 for dc_k in dc_candidates],
    })
    y.setRecords(y_fix)

    y.fx[k] = y[k]

    stoch_model.solve()

    exp_cost = float(z_expected.records["level"].iloc[0])
    scen_costs = z_s.records[["s", "level"]].copy()
    scen_costs.columns = ["scenario", "cost"]
    scen_costs["dc"] = dc

    results_per_dc.append({
        "dc":       dc,
        "exp_cost": exp_cost,
        "scen_df":  scen_costs,
    })
    print(f"  {dc:<20}  {exp_cost:>20,.2f}")


    y.lo[k] = 0
    y.up[k] = 1


best = min(results_per_dc, key=lambda r: r["exp_cost"])
optimal_dc = best["dc"]

print("\n" + "-" * 44)
print(f"  *  OPTIMAL DC  ->  {optimal_dc}  ({best['exp_cost']:,.2f} EUR)")
print("-" * 44)

# SOLUTION WITH THE OPTIMAL DC
y_opt = pd.DataFrame({
    "k":     dc_candidates,
    "level": [1.0 if dc_k == optimal_dc else 0.0 for dc_k in dc_candidates],
})
y.setRecords(y_opt)
y.fx[k] = y[k]
stoch_model.solve()
y.lo[k] = 0
y.up[k] = 1

#  Output matrices
scenario_order = ["lower", "low", "basic", "high", "higher"]
prob_dict      = dict(zip(scenario_order, [0.05, 0.15, 0.60, 0.15, 0.05]))

# COST PER SCENARIO
print("\n" + "="*70)
print(" COST PER SCENARIO")
print("="*70)

cost_df = z_s.records[["s", "level"]].copy()
cost_df.columns = ["Scenario", "Total Cost (EUR)"]
cost_df = cost_df.set_index("Scenario").reindex(scenario_order)
cost_df["Total Cost (EUR)"] = cost_df["Total Cost (EUR)"].map("{:,.2f}".format)
print(cost_df.to_string())

# Weighted Scenario Contribution
print("\n" + "="*70)
print("WEIGHTED SCENARIO CONTRIBUTION")
print("="*70)

wt_df = z_s.records[["s", "level"]].copy()
wt_df.columns = ["Scenario", "Cost (EUR)"]
wt_df["Probability"]      = wt_df["Scenario"].map(prob_dict)
wt_df["Weighted Cost (EUR)"] = wt_df["Cost (EUR)"] * wt_df["Probability"]
wt_df = wt_df.set_index("Scenario").reindex(scenario_order)
for col in ["Cost (EUR)", "Weighted Cost (EUR)"]:
    wt_df[col] = wt_df[col].map("{:,.2f}".format)
print(wt_df.to_string())
expected_val = sum(
    z_s.records.set_index("s").loc[sc, "level"] * prob_dict[sc]
    for sc in scenario_order
)
print(f"\n  Expected Total Cost = {expected_val:,.2f} EUR")

# Production per scenario
print("\n" + "="*70)
print("  PRODUCTION PER SCENARIO ")
print(f"  DC: {optimal_dc}")
print("="*70)

x_sel = x_s.records[
    (x_s.records["k"] == optimal_dc) &
    (x_s.records["level"] > 0)
].copy()

prod_table = x_sel.pivot_table(
    index="i",
    columns="s",
    values="level",
    aggfunc="sum",
    fill_value=0,
)
prod_table = prod_table.reindex(columns=scenario_order).fillna(0).round(0).astype(int)
prod_table.index.name   = "Product"
prod_table.columns.name = "Scenario"
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(prod_table.to_string())

# Customer demand
print("\n" + "="*70)
print(" Customer Demand")
print(f"  (rows=products x customers, cols=scenarios,  DC: {optimal_dc})")
print("="*70)

w_sel = w_s.records[
    (w_s.records["k"] == optimal_dc) &
    (w_s.records["level"] > 0)
].copy()

cust_table = w_sel.pivot_table(
    index=["i", "l"],
    columns="s",
    values="level",
    aggfunc="sum",
    fill_value=0,
)
cust_table = cust_table.reindex(columns=scenario_order).fillna(0).round(0).astype(int)
cust_table.index.names  = ["Product", "Customer Zone"]
cust_table.columns.name = "Scenario"
pd.set_option("display.max_rows", None)
print(cust_table.to_string())

# Summary: expected cost per DC
print("\n" + "="*70)
print("  SUMMARY: Expected Cost Ranking across all DC Candidates")
print("="*70)
ranking = pd.DataFrame([
    {"DC": r["dc"], "Expected Cost (EUR)": r["exp_cost"]}
    for r in results_per_dc
]).sort_values("Expected Cost (EUR)").reset_index(drop=True)
ranking.index += 1
ranking["Expected Cost (EUR)"] = ranking["Expected Cost (EUR)"].map("{:,.2f}".format)
print(ranking.to_string())
print(f"\n  Optimal DC: {optimal_dc}")
