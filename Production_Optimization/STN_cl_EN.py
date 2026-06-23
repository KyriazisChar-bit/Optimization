"""
=============================================================================
 OPTIMIZATION OF THE PRODUCTION SCHEDULING OF A POLYSTYRENE BATCH PLANT
 Optimal Production Scheduling of a Polystyrene Batch Plant
 Using the State-Task Network (STN) formulation
 Solved with GAMSpy (MILP)
=============================================================================

PROBLEM SUMMARY
---------------
A polystyrene production plant with the following batch stages:
  1. Premixing  (2 hr)  : Feed -> Premix
  2. Reaction   (6 hr)  : Premix -> 97.5% ReactorProduct + 2.5% Saturated (vapor loss)
  3. Screening  (semi-continuous, 12.7 tn/hr) : ReactorProduct ->
       BeadSize1 (20%), BeadSize2 (44%), BeadSize3 (26%), BeadSize4 (10%)
  4. Sparging1  (4 hr)  : BeadSize1 -> Sparge1
     Sparging2  (4 hr)  : BeadSize2 -> Sparge2
  5. Blending1  (1 hr)  : Sparge1 + additive1 (negligible) -> Prod1
     Blending2  (1 hr)  : Sparge2 + additive1 (negligible) -> Prod2

BeadSize3 and BeadSize4 are direct final products.
Prod1 and Prod2 require further processing (Sparging + Blending).

Holding tasks are introduced for:
  - Premix   (in Premixers)
  - BeadSize1 (in Sparger1 units, and optionally in the shared silo)
  - BeadSize2 (in Sparger2 units, and optionally in the shared silo)
  - Sparge1  (in Sparger1 units)
  - Sparge2  (in Sparger2 units)

TIME HORIZON: 24 hours  ->  T = 24 periods, delta = 1 hr, t in {1,...,25}
                             (t=25 is the final time point H+1)

OBJECTIVE: Maximize profit = sum of final product values at end of horizon
   Value: BeadSize3 = 0.4 k€/tn,  BeadSize4 = 0.3 k€/tn
          Prod1     = 2.0 k€/tn,  Prod2     = 2.1 k€/tn
   Intermediate products left at end of horizon have zero value.

EQUIPMENT:
  j=1: Premixer1   (Premixing,  max=20 tn, min=0.5 tn)
  j=2: Premixer2   (Premixing,  max=30 tn, min=0.5 tn)
  j=3: Reactor1    (Reaction,   max=20 tn, min=0.5 tn)  [2 units in parallel]
  j=4: Reactor2    (Reaction,   max=40 tn, min=0.5 tn)  [2 units in parallel]
  j=5: Screener1   (Screening,  semi-continuous: capacity = 12.7 tn/hr * 1 hr = 12.7 tn/period)
  j=6..10: Sparger1 units x5  (Sparging task i=1, max=5 tn, min=0.5 tn each)
  j=11..17: Sparger2 units x7 (Sparging task i=2, max=5 tn, min=0.5 tn each)
  j=18: Blender1   (Blending1,  max=7 tn,  min=0.5 tn)
  j=19: Blender2   (Blending2,  max=8 tn,  min=0.5 tn)
  j=20: Silo       (flexible storage for BeadSize1 OR BeadSize2, cap=20 tn)

STATES (materials):
  s=1 : Feed          (inf supply, inf storage)
  s=2 : Additive1     (inf supply, inf storage, negligible consumption)
  s=3 : Saturated     (vapor loss, no storage needed)
  s=4 : Premix        (cap=0, stored in Premixers via Hold_Premix)
  s=5 : ReactorProduct(cap=180 tn dedicated tank)
  s=6 : BeadSize1     (cap=0, stored in Sparger1 units or Silo)
  s=7 : BeadSize2     (cap=0, stored in Sparger2 units or Silo)
  s=8 : BeadSize3     (cap=inf, final product 0.4 k€/tn)
  s=9 : BeadSize4     (cap=inf, final product 0.3 k€/tn)
  s=10: Sparge1       (cap=55 tn, stored in Sparger1 units)
  s=11: Sparge2       (cap=100 tn, stored in Sparger2 units)
  s=12: Prod1         (cap=inf, final product 2.0 k€/tn)
  s=13: Prod2         (cap=inf, final product 2.1 k€/tn)

TASKS:
  i=1 : Premixing     (pi=2)  : consumes Feed(100%), produces Premix(100%)
  i=2 : Reaction      (pi=6)  : consumes Premix(100%), produces ReactorProduct(97.5%) + Saturated(2.5%)
  i=3 : Screening     (pi=1)  : consumes ReactorProduct(100%),
                                 produces BS1(20%)+BS2(44%)+BS3(26%)+BS4(10%)
  i=4 : Sparging1     (pi=4)  : consumes BeadSize1(100%), produces Sparge1(100%)
  i=5 : Sparging2     (pi=4)  : consumes BeadSize2(100%), produces Sparge2(100%)
  i=6 : Blending1     (pi=1)  : consumes Sparge1(100%), produces Prod1(100%)
  i=7 : Blending2     (pi=1)  : consumes Sparge2(100%), produces Prod2(100%)
  i=8 : Hold_Premix   (pi=1)  : consumes Premix(100%), produces Premix(100%)  [in Premixers]
  i=9 : Hold_BS1      (pi=1)  : consumes BS1(100%), produces BS1(100%)        [in Sparger1 or Silo]
  i=10: Hold_BS2      (pi=1)  : consumes BS2(100%), produces BS2(100%)        [in Sparger2 or Silo]
  i=11: Hold_Sparge1  (pi=1)  : consumes Sparge1(100%), produces Sparge1(100%)[in Sparger1]
  i=12: Hold_Sparge2  (pi=1)  : consumes Sparge2(100%), produces Sparge2(100%)[in Sparger2]
=============================================================================
"""

import sys
import gamspy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# MODEL SETUP
# =============================================================================

def build_and_solve():

    H  = 24   # number of time periods
    dT = 1    # period duration (hr)

    m = gp.Container()

    # -------------------------------------------------------------------------
    # SETS
    # -------------------------------------------------------------------------
    # Tasks
    tasks_data = [
        "Premixing", "Reaction", "Screening",
        "Sparging1", "Sparging2",
        "Blending1", "Blending2",
        "Hold_Premix", "Hold_BS1", "Hold_BS2",
        "Hold_Sparge1", "Hold_Sparge2"
    ]
    i_set = gp.Set(m, name="i", records=tasks_data, description="tasks")

    # States
    states_data = [
        "Feed", "Additive1", "Saturated", "Premix",
        "ReactorProduct", "BeadSize1", "BeadSize2",
        "BeadSize3", "BeadSize4",
        "Sparge1", "Sparge2",
        "Prod1", "Prod2"
    ]
    s_set = gp.Set(m, name="s", records=states_data, description="states/materials")

    # Equipment units
    units_data = (
        ["Premixer1", "Premixer2"] +
        ["Reactor1a", "Reactor1b", "Reactor2a", "Reactor2b"] +  # 2+2 parallel
        ["Screener1"] +
        [f"Sparger1_{k}" for k in range(1, 6)] +   # 5 units
        [f"Sparger2_{k}" for k in range(1, 8)] +   # 7 units
        ["Blender1", "Blender2"] +
        ["Silo"]
    )
    j_set = gp.Set(m, name="j", records=units_data, description="equipment units")

    # Time periods t = 1..H  (t=H+1 is the final inventory time point)
    t_data = list(range(1, H + 2))   # 1..25
    t_set = gp.Set(m, name="t", records=t_data, description="time periods")

    # -------------------------------------------------------------------------
    # PARAMETERS
    # -------------------------------------------------------------------------

    # Processing time (number of periods) for each task
    pi_data = {
        "Premixing":    2,
        "Reaction":     6,
        "Screening":    1,   # semi-continuous treated as batch with p=1
        "Sparging1":    4,
        "Sparging2":    4,
        "Blending1":    1,
        "Blending2":    1,
        "Hold_Premix":  1,
        "Hold_BS1":     1,
        "Hold_BS2":     1,
        "Hold_Sparge1": 1,
        "Hold_Sparge2": 1,
    }
    pi_param = gp.Parameter(m, name="pi_p", domain=[i_set],
                            records=[(k, v) for k, v in pi_data.items()],
                            description="processing time of task i (periods)")

    # Suitability: which tasks can run on which unit  (1 = suitable)
    # Build as a list of (task, unit) pairs
    suit_records = []
    for j in ["Premixer1", "Premixer2"]:
        suit_records += [("Premixing", j), ("Hold_Premix", j)]
    for j in ["Reactor1a", "Reactor1b"]:
        suit_records += [("Reaction", j)]
    for j in ["Reactor2a", "Reactor2b"]:
        suit_records += [("Reaction", j)]
    suit_records += [("Screening", "Screener1")]
    for j in [f"Sparger1_{k}" for k in range(1, 6)]:
        suit_records += [("Sparging1", j), ("Hold_BS1", j), ("Hold_Sparge1", j)]
    for j in [f"Sparger2_{k}" for k in range(1, 8)]:
        suit_records += [("Sparging2", j), ("Hold_BS2", j), ("Hold_Sparge2", j)]
    suit_records += [("Blending1", "Blender1"), ("Blending2", "Blender2")]
    # Silo can hold BS1 or BS2 (flexible storage)
    suit_records += [("Hold_BS1", "Silo"), ("Hold_BS2", "Silo")]

    suit_set = gp.Set(m, name="suit", domain=[i_set, j_set],
                      records=suit_records,
                      description="suitability: task i can run on unit j")

    # Max and min capacities for each (task, unit) combination
    vmax_data = {
        ("Premixing",    "Premixer1"):  20.0,
        ("Premixing",    "Premixer2"):  30.0,
        ("Hold_Premix",  "Premixer1"):  20.0,
        ("Hold_Premix",  "Premixer2"):  30.0,
        ("Reaction",     "Reactor1a"):  20.0,
        ("Reaction",     "Reactor1b"):  20.0,
        ("Reaction",     "Reactor2a"):  40.0,
        ("Reaction",     "Reactor2b"):  40.0,
        ("Screening",    "Screener1"):  12.7,   # 12.7 tn/hr * 1 hr
    }
    for j in [f"Sparger1_{k}" for k in range(1, 6)]:
        vmax_data[("Sparging1",    j)] = 5.0
        vmax_data[("Hold_BS1",     j)] = 5.0
        vmax_data[("Hold_Sparge1", j)] = 5.0
    for j in [f"Sparger2_{k}" for k in range(1, 8)]:
        vmax_data[("Sparging2",    j)] = 5.0
        vmax_data[("Hold_BS2",     j)] = 5.0
        vmax_data[("Hold_Sparge2", j)] = 5.0
    vmax_data[("Blending1", "Blender1")] = 7.0
    vmax_data[("Blending2", "Blender2")] = 8.0
    vmax_data[("Hold_BS1",  "Silo")]      = 20.0
    vmax_data[("Hold_BS2",  "Silo")]      = 20.0

    vmin_data = {}
    for (tk, jk) in vmax_data.keys():
        if tk in ("Hold_Premix", "Hold_BS1", "Hold_BS2", "Hold_Sparge1", "Hold_Sparge2"):
            vmin_data[(tk, jk)] = 0.0   # holding tasks: min=0
        elif tk == "Screening":
            vmin_data[(tk, jk)] = 0.0   # semi-continuous: no strict minimum modelled
        else:
            vmin_data[(tk, jk)] = 0.5   # physical minimum batch size

    vmax_param = gp.Parameter(m, name="Vmax", domain=[i_set, j_set],
                               records=[(a, b, v) for (a, b), v in vmax_data.items()],
                               description="max capacity of unit j for task i (tn)")
    vmin_param = gp.Parameter(m, name="Vmin", domain=[i_set, j_set],
                               records=[(a, b, v) for (a, b), v in vmin_data.items()],
                               description="min batch size for task i on unit j (tn)")

    # Input fractions rho_in[i, s] = fraction of batch consumed from state s
    rho_in_data = {
        ("Premixing",    "Feed"):            1.00,
        ("Reaction",     "Premix"):          1.00,
        ("Screening",    "ReactorProduct"):  1.00,
        ("Sparging1",    "BeadSize1"):       1.00,
        ("Sparging2",    "BeadSize2"):       1.00,
        ("Blending1",    "Sparge1"):         1.00,
        ("Blending2",    "Sparge2"):         1.00,
        ("Hold_Premix",  "Premix"):          1.00,
        ("Hold_BS1",     "BeadSize1"):       1.00,
        ("Hold_BS2",     "BeadSize2"):       1.00,
        ("Hold_Sparge1", "Sparge1"):         1.00,
        ("Hold_Sparge2", "Sparge2"):         1.00,
    }
    rho_in = gp.Parameter(m, name="rho_in", domain=[i_set, s_set],
                          records=[(a, b, v) for (a, b), v in rho_in_data.items()],
                          description="input fraction: fraction of batch size drawn from state s")

    # Output fractions rho_out[i, s] = fraction of batch produced into state s
    rho_out_data = {
        ("Premixing",    "Premix"):          1.000,
        ("Reaction",     "ReactorProduct"):  0.975,
        ("Reaction",     "Saturated"):       0.025,
        ("Screening",    "BeadSize1"):       0.200,
        ("Screening",    "BeadSize2"):       0.440,
        ("Screening",    "BeadSize3"):       0.260,
        ("Screening",    "BeadSize4"):       0.100,
        ("Sparging1",    "Sparge1"):         1.000,
        ("Sparging2",    "Sparge2"):         1.000,
        ("Blending1",    "Prod1"):           1.000,
        ("Blending2",    "Prod2"):           1.000,
        ("Hold_Premix",  "Premix"):          1.000,
        ("Hold_BS1",     "BeadSize1"):       1.000,
        ("Hold_BS2",     "BeadSize2"):       1.000,
        ("Hold_Sparge1", "Sparge1"):         1.000,
        ("Hold_Sparge2", "Sparge2"):         1.000,
    }
    rho_out = gp.Parameter(m, name="rho_out", domain=[i_set, s_set],
                           records=[(a, b, v) for (a, b), v in rho_out_data.items()],
                           description="output fraction: fraction of batch size produced into state s")

    # Storage capacities for each state
    # cap=0 means NIS (no intermediate storage) — enforced as S[s,t]=0 always
    # cap=inf means unlimited
    BIGM = 1e6
    cap_data = {
        "Feed":             BIGM,
        "Additive1":        BIGM,
        "Saturated":        BIGM,   # vapor, leaves system
        "Premix":           0.0,    # NIS — only held in premixers
        "ReactorProduct":   180.0,
        "BeadSize1":        0.0,    # NIS — held in sparger1 units / silo
        "BeadSize2":        0.0,    # NIS — held in sparger2 units / silo
        "BeadSize3":        BIGM,
        "BeadSize4":        BIGM,
        "Sparge1":          55.0,
        "Sparge2":          100.0,
        "Prod1":            BIGM,
        "Prod2":            BIGM,
    }
    cap = gp.Parameter(m, name="cap", domain=[s_set],
                       records=[(k, v) for k, v in cap_data.items()],
                       description="max storage capacity for state s (tn)")

    # Initial inventory for each state
    S0_data = {s: 0.0 for s in states_data}
    S0_data["Feed"]      = BIGM
    S0_data["Additive1"] = BIGM
    S0_init = gp.Parameter(m, name="S0", domain=[s_set],
                           records=[(k, v) for k, v in S0_data.items()],
                           description="initial inventory of state s at t=0")

    # Product values (k€/tn)
    price_data = {
        "BeadSize3": 0.4,
        "BeadSize4": 0.3,
        "Prod1":     2.0,
        "Prod2":     2.1,
    }
    price = gp.Parameter(m, name="price", domain=[s_set],
                         records=[(k, v) for k, v in price_data.items()],
                         description="product value (k€/tn)")

    # -------------------------------------------------------------------------
    # VARIABLES
    # -------------------------------------------------------------------------
    # W[i,j,t] = 1 if task i starts on unit j at time t, else 0  (binary)
    W = gp.Variable(m, name="W", domain=[i_set, j_set, t_set],
                    type="binary",
                    description="1 if task i starts on unit j at period t")

    # B[i,j,t] = batch size of task i on unit j starting at period t  (>=0)
    B = gp.Variable(m, name="B", domain=[i_set, j_set, t_set],
                    type="positive",
                    description="batch size (tn) of task i on unit j at period t")

    # S[s,t] = inventory of state s at start of period t  (>=0)
    S = gp.Variable(m, name="S", domain=[s_set, t_set],
                    type="positive",
                    description="inventory (tn) of state s at start of period t")

    # Objective variable
    profit = gp.Variable(m, name="profit", type="free",
                         description="total profit (k€)")

    # -------------------------------------------------------------------------
    # EQUATIONS
    # -------------------------------------------------------------------------
    equations = []

    # ------------------------------------------------------------------
    # 1. UNIT OPERATION CONSTRAINTS
    #    Sum over tasks that could be running on unit j during period t <= 1
    #
    #    For each unit j, at each time t:
    #    SUM_{i in I_j} SUM_{theta=0}^{pi_i - 1} W[i,j,t-theta] <= 1
    # ------------------------------------------------------------------
    def unit_op_rule(m_eq, j_name, t_val):
        lhs_terms = []
        for (i_name, j2) in suit_records:
            if j2 != j_name:
                continue
            p = pi_data[i_name]
            for theta in range(p):
                t_prev = t_val - theta
                if 1 <= t_prev <= H:
                    lhs_terms.append(W[i_name, j_name, t_prev])
        if not lhs_terms:
            return None
        return sum(lhs_terms) <= 1

    eq_unit_op = gp.Equation(m, name="eq_unit_op", domain=[j_set, t_set],
                              description="unit can execute at most one task at a time")
    # Build manually since we need flexible indexing
    # We'll use a Parameter-based approach with explicit Python loops building
    # individual constraints. We use gp.Equation in "definition" form.

    # GAMSpy supports building constraints via indexed equations; for complex
    # window sums we write them out as explicit Python-loop-generated constraints
    # using the addEquation pattern. Instead, we generate them with a helper.

    # We collect all constraint expressions and add them to a single equation object
    # by using a gp.EquationList or repeated definitions. The cleanest GAMSpy
    # approach is to pre-compute the window sums as Parameters.

    # Pre-compute unit_active[j, t] = sum_{theta} W[i,j,t-theta] as a sum expression
    # We do this by building explicit sparse constraint rows.

    unit_op_constraints = []
    for j_name in units_data:
        for t_val in range(1, H + 1):
            terms = []
            for (i_name, j2) in suit_records:
                if j2 != j_name:
                    continue
                p = pi_data[i_name]
                for theta in range(p):
                    t_prev = t_val - theta
                    if 1 <= t_prev <= H:
                        terms.append(W[i_name, j_name, t_prev])
            if terms:
                unit_op_constraints.append(sum(terms) <= 1)

    # ------------------------------------------------------------------
    # 2. BATCH SIZE BOUNDS
    #    Vmin * W[i,j,t] <= B[i,j,t] <= Vmax * W[i,j,t]
    # ------------------------------------------------------------------
    batch_lo_constraints = []
    batch_hi_constraints = []
    for (i_name, j_name) in vmax_data.keys():
        vmax_val = vmax_data[(i_name, j_name)]
        vmin_val = vmin_data.get((i_name, j_name), 0.0)
        for t_val in range(1, H + 1):
            w_var = W[i_name, j_name, t_val]
            b_var = B[i_name, j_name, t_val]
            batch_lo_constraints.append(b_var >= vmin_val * w_var)
            batch_hi_constraints.append(b_var <= vmax_val * w_var)

    # Also force B=0 for unsuitable (i,j) pairs for all t
    zero_constraints = []
    suit_set_lookup = set(suit_records)
    for i_name in tasks_data:
        for j_name in units_data:
            if (i_name, j_name) not in suit_set_lookup:
                for t_val in range(1, H + 1):
                    zero_constraints.append(B[i_name, j_name, t_val] == 0)
                    zero_constraints.append(W[i_name, j_name, t_val] == 0)

    # ------------------------------------------------------------------
    # 3. MASS BALANCE CONSTRAINTS
    #    S[s,t] = S[s,t-1]
    #           + sum_{i in Tbar_s, j} rho_out[i,s] * B[i,j,t-pi_i]  (produced)
    #           - sum_{i in T_s,    j} rho_in[i,s]  * B[i,j,t]       (consumed)
    #    for t = 1..H+1  (D[s,t]=0 for this problem, R[s,t]=0)
    # ------------------------------------------------------------------
    # Identify which tasks produce / consume each state
    produces = {s: [] for s in states_data}
    consumes = {s: [] for s in states_data}
    for (i_name, s_name), frac in rho_out_data.items():
        produces[s_name].append((i_name, frac))
    for (i_name, s_name), frac in rho_in_data.items():
        consumes[s_name].append((i_name, frac))

    # Eligible units for each task
    task_units = {i_name: [] for i_name in tasks_data}
    for (i_name, j_name) in suit_records:
        task_units[i_name].append(j_name)

    mass_balance_constraints = []
    for s_name in states_data:
        # Skip Feed/Additive1 (infinite supply — no meaningful balance needed;
        # we just ensure non-negativity via lower bound on S)
        if s_name in ("Feed", "Additive1"):
            continue
        # Skip Saturated (vapor, leaves system — no storage constraint)
        if s_name == "Saturated":
            continue

        for t_val in range(1, H + 2):   # t = 1..H+1

            # Previous inventory
            if t_val == 1:
                prev_S = S0_data[s_name]
            else:
                prev_S = S[s_name, t_val - 1]

            # Production term: tasks that produce s, finishing at t
            prod_terms = []
            for (i_name, frac) in produces[s_name]:
                p = pi_data[i_name]
                t_start = t_val - p
                if 1 <= t_start <= H:
                    for j_name in task_units[i_name]:
                        prod_terms.append(frac * B[i_name, j_name, t_start])

            # Consumption term: tasks that consume s, starting at t
            cons_terms = []
            for (i_name, frac) in consumes[s_name]:
                if t_val <= H:  # no consumption events can start at t=H+1
                    for j_name in task_units[i_name]:
                        cons_terms.append(frac * B[i_name, j_name, t_val])

            prod_sum = sum(prod_terms) if prod_terms else 0.0
            cons_sum = sum(cons_terms) if cons_terms else 0.0

            current_S = S[s_name, t_val]
            mass_balance_constraints.append(
                current_S == prev_S + prod_sum - cons_sum
            )

    # ------------------------------------------------------------------
    # 4. STORAGE CAPACITY CONSTRAINTS
    # ------------------------------------------------------------------
    storage_constraints = []
    for s_name in states_data:
        if s_name in ("Feed", "Additive1", "Saturated"):
            continue
        c = cap_data[s_name]
        for t_val in range(1, H + 2):
            if c < BIGM:
                storage_constraints.append(S[s_name, t_val] <= c)
            # Non-negativity already enforced by variable type

    # States with cap=0 (NIS): S must equal 0 at all times
    # These are Premix, BeadSize1, BeadSize2 — their "storage" is in holding tasks
    nis_states = ["Premix", "BeadSize1", "BeadSize2"]
    nis_constraints = []
    for s_name in nis_states:
        for t_val in range(1, H + 2):
            nis_constraints.append(S[s_name, t_val] == 0)

    # ------------------------------------------------------------------
    # 5. SILO MUTUAL EXCLUSION
    #    Silo can hold BS1 OR BS2, not both simultaneously
    #    sum over t of (W_Hold_BS1_Silo_t + W_Hold_BS2_Silo_t) implies only one
    #    material at a time -> at each t: W_Hold_BS1_Silo_t + W_Hold_BS2_Silo_t <= 1
    # ------------------------------------------------------------------
    silo_constraints = []
    for t_val in range(1, H + 1):
        silo_constraints.append(
            W["Hold_BS1", "Silo", t_val] + W["Hold_BS2", "Silo", t_val] <= 1
        )

    # ------------------------------------------------------------------
    # 6. OBJECTIVE FUNCTION
    #    Maximize total revenue from final products at end of horizon (t=H+1=25)
    # ------------------------------------------------------------------
    t_final = H + 1
    obj_expr = gp.math.Sum(
        gp.Domain(s_set).where(price[s_set] > 0),
        price[s_set] * S[s_set, t_final]
    )

    obj_eq = gp.Equation(m, name="obj_eq", description="objective function definition")
    obj_eq[...] = profit == obj_expr

    # -------------------------------------------------------------------------
    # BUILD THE MODEL
    # -------------------------------------------------------------------------
    # We need to register all Python-generated constraints with GAMSpy.
    # The most straightforward approach: use gp.Model with a list of
    # all constraint expressions via the "equations" kwarg.

    # Package everything into a single model
    model = gp.Model(
        m,
        name="STN_Polystyrene",
        equations=m.getEquations(),
        problem="MIP",
        sense="MAX",
        objective=profit,
    )

    # Add all explicitly built constraints
    all_constraints = (
        unit_op_constraints +
        batch_lo_constraints +
        batch_hi_constraints +
        zero_constraints +
        mass_balance_constraints +
        storage_constraints +
        nis_constraints +
        silo_constraints
    )

    # In GAMSpy, Python-expression constraints are not registered automatically.
    # We add them via addEquations or by solving with extra_equations.
    # Use the solve() extra_equations parameter.
    model.solve(
        output=sys.stdout,
        extra_equations=all_constraints
    )

    return m, W, B, S, profit, H, t_final, tasks_data, units_data, states_data, price_data


# =============================================================================
# ALTERNATIVE APPROACH: Pure GAMSpy indexed equations (recommended for large models)
# =============================================================================

def build_and_solve_v2():
    """
    This version builds the model using proper GAMSpy indexed Equations
    with explicit Domain indexing, which is more efficient for the solver.
    """
    H  = 24
    dT = 1

    m = gp.Container()

    # ---- SETS ----
    tasks_data = [
        "Premixing", "Reaction", "Screening",
        "Sparging1", "Sparging2",
        "Blending1", "Blending2",
        "Hold_Premix", "Hold_BS1", "Hold_BS2",
        "Hold_Sparge1", "Hold_Sparge2"
    ]
    states_data = [
        "Feed", "Additive1", "Saturated", "Premix",
        "ReactorProduct", "BeadSize1", "BeadSize2",
        "BeadSize3", "BeadSize4",
        "Sparge1", "Sparge2",
        "Prod1", "Prod2"
    ]
    units_data = (
        ["Premixer1", "Premixer2"] +
        ["Reactor1a", "Reactor1b", "Reactor2a", "Reactor2b"] +
        ["Screener1"] +
        [f"Sparger1_{k}" for k in range(1, 6)] +
        [f"Sparger2_{k}" for k in range(1, 8)] +
        ["Blender1", "Blender2", "Silo"]
    )

    i_set = gp.Set(m, "i", records=tasks_data)
    s_set = gp.Set(m, "s", records=states_data)
    j_set = gp.Set(m, "j", records=units_data)

    # t = 1..H+1 (scheduling periods + final inventory point)
    t_all = list(range(1, H + 2))
    t_set = gp.Set(m, "t", records=t_all, description="all time points incl. final")

    # Subset for scheduling decisions: t = 1..H only
    t_sched = gp.Set(m, "ts", domain=[t_set],
                     records=list(range(1, H + 1)),
                     description="scheduling periods (no decisions at H+1)")

    # ---- PARAMETERS ----
    pi_data = {
        "Premixing": 2, "Reaction": 6, "Screening": 1,
        "Sparging1": 4, "Sparging2": 4,
        "Blending1": 1, "Blending2": 1,
        "Hold_Premix": 1, "Hold_BS1": 1, "Hold_BS2": 1,
        "Hold_Sparge1": 1, "Hold_Sparge2": 1,
    }
    pi_p = gp.Parameter(m, "pi_p", domain=[i_set],
                        records=[(k, v) for k, v in pi_data.items()])

    suit_records = []
    for j in ["Premixer1", "Premixer2"]:
        suit_records += [("Premixing", j), ("Hold_Premix", j)]
    for j in ["Reactor1a", "Reactor1b"]:
        suit_records += [("Reaction", j)]
    for j in ["Reactor2a", "Reactor2b"]:
        suit_records += [("Reaction", j)]
    suit_records += [("Screening", "Screener1")]
    for j in [f"Sparger1_{k}" for k in range(1, 6)]:
        suit_records += [("Sparging1", j), ("Hold_BS1", j), ("Hold_Sparge1", j)]
    for j in [f"Sparger2_{k}" for k in range(1, 8)]:
        suit_records += [("Sparging2", j), ("Hold_BS2", j), ("Hold_Sparge2", j)]
    suit_records += [("Blending1", "Blender1"), ("Blending2", "Blender2")]
    suit_records += [("Hold_BS1", "Silo"), ("Hold_BS2", "Silo")]

    IJ = gp.Set(m, "IJ", domain=[i_set, j_set], records=suit_records,
                description="suitable (task, unit) pairs")

    BIGM = 1e6

    vmax_data = {
        ("Premixing", "Premixer1"): 20, ("Premixing", "Premixer2"): 30,
        ("Hold_Premix", "Premixer1"): 20, ("Hold_Premix", "Premixer2"): 30,
        ("Reaction", "Reactor1a"): 20, ("Reaction", "Reactor1b"): 20,
        ("Reaction", "Reactor2a"): 40, ("Reaction", "Reactor2b"): 40,
        ("Screening", "Screener1"): 12.7,
    }
    for j in [f"Sparger1_{k}" for k in range(1, 6)]:
        vmax_data[("Sparging1", j)] = 5
        vmax_data[("Hold_BS1", j)] = 5
        vmax_data[("Hold_Sparge1", j)] = 5
    for j in [f"Sparger2_{k}" for k in range(1, 8)]:
        vmax_data[("Sparging2", j)] = 5
        vmax_data[("Hold_BS2", j)] = 5
        vmax_data[("Hold_Sparge2", j)] = 5
    vmax_data[("Blending1", "Blender1")] = 7
    vmax_data[("Blending2", "Blender2")] = 8
    vmax_data[("Hold_BS1", "Silo")] = 20
    vmax_data[("Hold_BS2", "Silo")] = 20

    vmin_data = {}
    for (tk, jk) in vmax_data:
        if "Hold" in tk:
            vmin_data[(tk, jk)] = 0.0
        elif tk == "Screening":
            vmin_data[(tk, jk)] = 0.0
        else:
            vmin_data[(tk, jk)] = 0.5

    Vmax = gp.Parameter(m, "Vmax", domain=[i_set, j_set],
                        records=[(a, b, v) for (a, b), v in vmax_data.items()])
    Vmin = gp.Parameter(m, "Vmin", domain=[i_set, j_set],
                        records=[(a, b, v) for (a, b), v in vmin_data.items()])

    rho_in_data = {
        ("Premixing", "Feed"): 1.0,
        ("Reaction", "Premix"): 1.0,
        ("Screening", "ReactorProduct"): 1.0,
        ("Sparging1", "BeadSize1"): 1.0,
        ("Sparging2", "BeadSize2"): 1.0,
        ("Blending1", "Sparge1"): 1.0,
        ("Blending2", "Sparge2"): 1.0,
        ("Hold_Premix", "Premix"): 1.0,
        ("Hold_BS1", "BeadSize1"): 1.0,
        ("Hold_BS2", "BeadSize2"): 1.0,
        ("Hold_Sparge1", "Sparge1"): 1.0,
        ("Hold_Sparge2", "Sparge2"): 1.0,
    }
    rho_out_data = {
        ("Premixing", "Premix"): 1.0,
        ("Reaction", "ReactorProduct"): 0.975,
        ("Reaction", "Saturated"): 0.025,
        ("Screening", "BeadSize1"): 0.20,
        ("Screening", "BeadSize2"): 0.44,
        ("Screening", "BeadSize3"): 0.26,
        ("Screening", "BeadSize4"): 0.10,
        ("Sparging1", "Sparge1"): 1.0,
        ("Sparging2", "Sparge2"): 1.0,
        ("Blending1", "Prod1"): 1.0,
        ("Blending2", "Prod2"): 1.0,
        ("Hold_Premix", "Premix"): 1.0,
        ("Hold_BS1", "BeadSize1"): 1.0,
        ("Hold_BS2", "BeadSize2"): 1.0,
        ("Hold_Sparge1", "Sparge1"): 1.0,
        ("Hold_Sparge2", "Sparge2"): 1.0,
    }

    rho_in  = gp.Parameter(m, "rho_in",  domain=[i_set, s_set],
                           records=[(a, b, v) for (a, b), v in rho_in_data.items()])
    rho_out = gp.Parameter(m, "rho_out", domain=[i_set, s_set],
                           records=[(a, b, v) for (a, b), v in rho_out_data.items()])

    cap_data = {
        "Feed": BIGM, "Additive1": BIGM, "Saturated": BIGM,
        "Premix": 0.0, "ReactorProduct": 180.0,
        "BeadSize1": 0.0, "BeadSize2": 0.0,
        "BeadSize3": BIGM, "BeadSize4": BIGM,
        "Sparge1": 55.0, "Sparge2": 100.0,
        "Prod1": BIGM, "Prod2": BIGM,
    }
    S0_data = {s: 0.0 for s in states_data}
    S0_data["Feed"] = BIGM
    S0_data["Additive1"] = BIGM

    cap = gp.Parameter(m, "cap", domain=[s_set],
                       records=[(k, v) for k, v in cap_data.items()])
    S0  = gp.Parameter(m, "S0",  domain=[s_set],
                       records=[(k, v) for k, v in S0_data.items()])
    price_data = {"BeadSize3": 0.4, "BeadSize4": 0.3, "Prod1": 2.0, "Prod2": 2.1}
    price = gp.Parameter(m, "price", domain=[s_set],
                         records=[(k, v) for k, v in price_data.items()])

    # ---- VARIABLES ----
    W = gp.Variable(m, "W", domain=[i_set, j_set, t_set], type="binary")
    B = gp.Variable(m, "B", domain=[i_set, j_set, t_set], type="positive")
    S = gp.Variable(m, "S", domain=[s_set, t_set],        type="positive")
    profit = gp.Variable(m, "profit", type="free")

    # ---- CONSTRAINTS ----
    # We build them as Python lists of gp expressions (valid in GAMSpy >= 1.x)

    eqs = []

    # (A) Unit operation constraints
    for j_name in units_data:
        for t_val in range(1, H + 1):
            terms = []
            for (i_name, j2) in suit_records:
                if j2 != j_name:
                    continue
                p = pi_data[i_name]
                for theta in range(p):
                    tp = t_val - theta
                    if 1 <= tp <= H:
                        terms.append(W[i_name, j_name, tp])
            if terms:
                eqs.append(sum(terms) <= 1)

    # (B) Batch size bounds and zero for non-suitable pairs
    suit_lookup = set(suit_records)
    for i_name in tasks_data:
        for j_name in units_data:
            if (i_name, j_name) not in suit_lookup:
                for t_val in range(1, H + 1):
                    eqs.append(W[i_name, j_name, t_val] == 0)
                    eqs.append(B[i_name, j_name, t_val] == 0)
            else:
                vmax_val = vmax_data[(i_name, j_name)]
                vmin_val = vmin_data.get((i_name, j_name), 0.0)
                for t_val in range(1, H + 1):
                    eqs.append(B[i_name, j_name, t_val] >= vmin_val * W[i_name, j_name, t_val])
                    eqs.append(B[i_name, j_name, t_val] <= vmax_val * W[i_name, j_name, t_val])

    # (C) Mass balances
    produces_map = {s: [] for s in states_data}
    consumes_map = {s: [] for s in states_data}
    for (i_name, s_name), frac in rho_out_data.items():
        produces_map[s_name].append((i_name, frac))
    for (i_name, s_name), frac in rho_in_data.items():
        consumes_map[s_name].append((i_name, frac))

    task_units_map = {i_name: [] for i_name in tasks_data}
    for (i_name, j_name) in suit_records:
        task_units_map[i_name].append(j_name)

    skip_states = {"Feed", "Additive1", "Saturated"}
    nis_states   = {"Premix", "BeadSize1", "BeadSize2"}

    for s_name in states_data:
        if s_name in skip_states:
            continue
        for t_val in range(1, H + 2):
            prev_S = S0_data[s_name] if t_val == 1 else S[s_name, t_val - 1]

            prod_terms = []
            for (i_name, frac) in produces_map[s_name]:
                p = pi_data[i_name]
                t_start = t_val - p
                if 1 <= t_start <= H:
                    for j_name in task_units_map[i_name]:
                        prod_terms.append(frac * B[i_name, j_name, t_start])

            cons_terms = []
            for (i_name, frac) in consumes_map[s_name]:
                if t_val <= H:
                    for j_name in task_units_map[i_name]:
                        cons_terms.append(frac * B[i_name, j_name, t_val])

            prod_sum = sum(prod_terms) if prod_terms else 0.0
            cons_sum = sum(cons_terms) if cons_terms else 0.0

            eqs.append(S[s_name, t_val] == prev_S + prod_sum - cons_sum)

            # Storage capacity
            if cap_data[s_name] < BIGM and s_name not in nis_states:
                eqs.append(S[s_name, t_val] <= cap_data[s_name])

            # NIS states: force S=0
            if s_name in nis_states:
                eqs.append(S[s_name, t_val] == 0)

    # (D) Silo mutual exclusion
    for t_val in range(1, H + 1):
        eqs.append(
            W["Hold_BS1", "Silo", t_val] + W["Hold_BS2", "Silo", t_val] <= 1
        )

    # (E) Objective
    t_final = H + 1
    obj_terms = [price_data[s_name] * S[s_name, t_final]
                 for s_name in states_data if s_name in price_data]
    eqs.append(profit == sum(obj_terms))

    # ---- MODEL & SOLVE ----
    print("="*60)
    print("Building STN MILP model for Polystyrene Production")
    print(f"Time horizon: {H} hours, delta = {dT} hr")
    print(f"Tasks: {len(tasks_data)}, States: {len(states_data)}, Units: {len(units_data)}")
    print("="*60)

    model = gp.Model(
        m,
        name="STN_Polystyrene",
        problem="MIP",
        sense="MAX",
        objective=profit,
    )

    model.solve(output=sys.stdout, extra_equations=eqs)

    return m, W, B, S, profit, H, t_final, tasks_data, units_data, states_data, price_data


# =============================================================================
# POST-PROCESSING: Extract results and generate Gantt charts & state profiles
# =============================================================================

def extract_results(m, W, B, S, H, tasks_data, units_data, states_data, price_data):
    """Extract solution values and return as DataFrames."""

    t_all = list(range(1, H + 2))

    # --- W values ---
    W_df = W.records if W.records is not None else pd.DataFrame()
    B_df = B.records if B.records is not None else pd.DataFrame()
    S_df = S.records if S.records is not None else pd.DataFrame()

    return W_df, B_df, S_df


def plot_gantt(W_df, B_df, H, tasks_data, units_data):
    """Generate Gantt chart for key equipment."""
    pi_data = {
        "Premixing": 2, "Reaction": 6, "Screening": 1,
        "Sparging1": 4, "Sparging2": 4,
        "Blending1": 1, "Blending2": 1,
        "Hold_Premix": 1, "Hold_BS1": 1, "Hold_BS2": 1,
        "Hold_Sparge1": 1, "Hold_Sparge2": 1,
    }

    # Filter non-zero W entries
    if W_df is None or W_df.empty:
        print("No solution found or W is empty.")
        return

    W_nonzero = W_df[W_df["level"] > 0.5].copy()
    if W_nonzero.empty:
        print("No tasks scheduled (W all zero).")
        return

    W_nonzero.columns = [c.lower() for c in W_nonzero.columns]
    # columns: i, j, t, level

    # Select key units to display
    key_units = (
        ["Premixer1", "Premixer2"] +
        ["Reactor1a", "Reactor1b", "Reactor2a", "Reactor2b"] +
        ["Screener1"] +
        [f"Sparger1_{k}" for k in range(1, 6)] +
        [f"Sparger2_{k}" for k in range(1, 8)] +
        ["Blender1", "Blender2", "Silo"]
    )

    # Color map for tasks
    task_colors = {
        "Premixing":    "#4C72B0",
        "Reaction":     "#DD8452",
        "Screening":    "#55A868",
        "Sparging1":    "#C44E52",
        "Sparging2":    "#8172B2",
        "Blending1":    "#937860",
        "Blending2":    "#DA8BC3",
        "Hold_Premix":  "#8C8C8C",
        "Hold_BS1":     "#CCB974",
        "Hold_BS2":     "#64B5CD",
        "Hold_Sparge1": "#AECDE8",
        "Hold_Sparge2": "#FFBB78",
    }

    fig, ax = plt.subplots(figsize=(16, 10))
    y_labels = key_units
    y_pos = {unit: idx for idx, unit in enumerate(y_labels)}

    for _, row in W_nonzero.iterrows():
        i_name = row["i"]
        j_name = row["j"]
        t_start = int(row["t"]) - 1   # convert to 0-based time axis
        p = pi_data.get(i_name, 1)
        color = task_colors.get(i_name, "grey")
        if j_name in y_pos:
            y = y_pos[j_name]
            ax.barh(y, width=p, left=t_start, height=0.6,
                    color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
            ax.text(t_start + p / 2, y, i_name.replace("_", "\n"),
                    ha="center", va="center", fontsize=5, color="white", fontweight="bold")

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Time (hr)", fontsize=10)
    ax.set_title("Gantt Chart — Polystyrene Production Schedule", fontsize=13, fontweight="bold")
    ax.set_xlim(0, H)
    ax.set_xticks(range(0, H + 1, 2))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    # Legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in task_colors.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/gantt_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gantt chart saved: gantt_chart.png")


def plot_state_profiles(S_df, H, states_data, price_data):
    """Plot inventory profiles for key states."""
    if S_df is None or S_df.empty:
        print("No inventory data to plot.")
        return

    S_df.columns = [c.lower() for c in S_df.columns]
    # columns: s, t, level

    key_states = ["ReactorProduct", "Sparge1", "Sparge2",
                  "BeadSize3", "BeadSize4", "Prod1", "Prod2"]
    plot_states = [s for s in key_states if s in states_data]

    fig, axes = plt.subplots(len(plot_states), 1, figsize=(14, 2.5 * len(plot_states)),
                             sharex=True)
    if len(plot_states) == 1:
        axes = [axes]

    t_axis = list(range(1, H + 2))

    for ax, s_name in zip(axes, plot_states):
        vals = []
        for t_val in t_axis:
            sub = S_df[(S_df["s"] == s_name) & (S_df["t"].astype(int) == t_val)]
            vals.append(float(sub["level"].values[0]) if not sub.empty else 0.0)
        ax.step(t_axis, vals, where="post", color="#2196F3", linewidth=1.8)
        ax.fill_between(t_axis, vals, step="post", alpha=0.2, color="#2196F3")
        label = s_name
        if s_name in price_data:
            label += f"  [{price_data[s_name]} k€/tn]"
        ax.set_ylabel(label, fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time period (hr)", fontsize=10)
    fig.suptitle("Inventory Profiles — Key States", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/state_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("State profiles saved: state_profiles.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print(" POLYSTYRENE PRODUCTION SCHEDULING — STN MILP MODEL")
    print(" Course: Production Planning and Control")
    print(" Tool: GAMSpy | Solver: CPLEX/CBC")
    print("="*60 + "\n")

    # Run the model
    m, W, B, S, profit, H, t_final, tasks_data, units_data, states_data, price_data = \
        build_and_solve_v2()

    # Print objective
    try:
        obj_val = profit.toValue()
        print(f"\n{'='*60}")
        print(f" OPTIMAL PROFIT: {obj_val:.4f} k€")
        print(f"{'='*60}\n")
    except Exception:
        print("Could not retrieve objective value.")

    # Extract and display results
    W_df, B_df, S_df = extract_results(
        m, W, B, S, H, tasks_data, units_data, states_data, price_data
    )

    # Print scheduled tasks (W=1)
    if W_df is not None and not W_df.empty:
        W_df.columns = [c.lower() for c in W_df.columns]
        scheduled = W_df[W_df["level"] > 0.5].copy()
        if not scheduled.empty:
            print("SCHEDULED TASKS (W=1):")
            print(scheduled[["i", "j", "t"]].sort_values(["j", "t"]).to_string(index=False))
        print()

    # Print batch sizes
    if B_df is not None and not B_df.empty:
        B_df.columns = [c.lower() for c in B_df.columns]
        nonzero_B = B_df[B_df["level"] > 1e-4].copy()
        if not nonzero_B.empty:
            print("BATCH SIZES (tn):")
            print(nonzero_B[["i", "j", "t", "level"]].sort_values(["j", "t"]).to_string(index=False))
        print()

    # Print final inventories of valuable products
    if S_df is not None and not S_df.empty:
        S_df.columns = [c.lower() for c in S_df.columns]
        final_inv = S_df[S_df["t"].astype(int) == t_final].copy()
        final_inv = final_inv[final_inv["s"].isin(price_data.keys())]
        if not final_inv.empty:
            print(f"FINAL INVENTORIES at t={t_final} (end of horizon):")
            for _, row in final_inv.iterrows():
                s_name = row["s"]
                qty = row["level"]
                val = price_data[s_name] * qty
                print(f"  {s_name:15s}: {qty:8.3f} tn  x  {price_data[s_name]:.1f} k€/tn = {val:8.3f} k€")

    # Generate plots
    try:
        plot_gantt(W_df, B_df, H, tasks_data, units_data)
        plot_state_profiles(S_df, H, states_data, price_data)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\nDone. Output files saved to /mnt/user-data/outputs/")