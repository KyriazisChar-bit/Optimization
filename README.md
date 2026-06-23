# 🧮 Operations Research & Optimization — Course Projects

Two industrial optimization problems modelled as Mixed-Integer Linear Programs (MILP), built in Python with GAMSPy and solved to optimality.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)  
**Course:** Production Planning & Control  
**Instructor:** Georgios Georgiadis, Assistant Professor  
**Author:** Kyriazis Charitopoulos  
**Semester:** Spring 2025–2026

---

## ⚙️ Tech Stack

- Python 3
- GAMSPy (GAMS algebraic modelling in Python) + MILP solver
- pandas, NumPy
- Matplotlib (Gantt charts, inventory profiles)

---

## 🏗️ The Projects

**Project 1 — Production & Distribution Network Design**  
A facility-location MILP: a manufacturer with two Greek factories must open exactly one of eight candidate distribution centers in Germany to serve ten customer zones at minimum total cost (production + installation + transport + handling). Extended into a stochastic version with five demand scenarios, minimising *expected* cost. **Optimal DC: Erfurt** — €4,412,384.92 deterministic, €4,414,280.32 expected. See `project-1-network-design/`.

**Project 2 — Polystyrene Batch-Plant Scheduling**  
A State-Task Network (STN) scheduling MILP: a five-stage polystyrene plant (premixing → reaction → screening → sparging → blending) with 21 units must produce a profit-maximizing 24-hour schedule, respecting batch durations, no-storage intermediates, equipment-holding logic, and a single-occupancy shared silo. **Maximum profit: 193.696 k€**; screening identified as the bottleneck. See `project-2-stn-scheduling/`.

---

## 🧠 Skills & Way of Thinking

**Mathematical modelling**  
Translating a word-based business problem into formal optimization: defining indices/sets, parameters, decision variables, constraints, and an objective. The central habit — asking "what is actually being *decided* here?" — is what separates a variable from a parameter.

**Mixed-Integer programming**  
Using binary variables to encode logical decisions (open a DC `y[k]`; start a task `W[i,j,t]`) and coupling them to continuous quantities (flows, batch sizes) through linking constraints like `Vmin·W ≤ B ≤ Vmax·W`.

**Constraint formulation**  
Expressing physics and business rules as linear inequalities — flow balance, capacity, demand satisfaction, mass conservation with yield/evaporation, big-M switching, and mutual exclusion — and recognising when a constraint is non-binding because a tighter one dominates.

**Optimization under uncertainty**  
Extending a deterministic model with a scenario index and probabilities to minimise *expected* cost: the robust choice isn't best on average, it's best across all weighted cases.

**Batch-process scheduling (STN)**  
Modelling a plant as tasks consuming/producing states on shared units over discrete time — handling semi-continuous vs. batch operations, multi-period equipment occupancy, intermediates with no storage, and shared-resource exclusivity.

**Reading a solution, not just solving**  
Identifying bottlenecks, explaining *why* the solver chose what it did, cross-checking output against inputs, using marginal/reduced costs (the €96,605 second-best gap), and communicating schedules visually via Gantt charts.

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `Network_Optimization_EN_2.py` | Facility-location MILP (deterministic + stochastic) |
| `Network_Optimization_Report_EN.pdff` | Mathematical model, methodology, results |
| `STN_cl_EN.py` | Polystyrene STN scheduling model + plotting |
| `project-2-stn-scheduling/report.pdf` | STN diagram, model, schedule analysis |
| `*/docs/assignment.pdf` | Original assignment briefs (reference) |

---

## 🚀 Quick Start

**Requirements:** Python 3.8+, a working GAMS/GAMSPy installation

```bash
git clone https://github.com/<your-username>/optimization-portfolio.git
cd optimization-portfolio
pip install gamspy pandas numpy matplotlib
```

Then enter a project folder and run its script:

```bash
cd project-1-network-design
python Network_Optimization_EN_2.py
```

A free GAMS community/demo license covers models of this size.

---

## 📊 Results at a Glance

| Project | Type | Objective | Outcome |
|---------|------|-----------|---------|
| 1 — Network Design | Facility location | Minimise cost | Erfurt — €4,412,384.92 (det.) / €4,414,280.32 (exp.) |
| 2 — STN Scheduling | Batch scheduling | Maximise profit | 193.696 k€ — screening is the bottleneck |

---

## 👥 Author

| Field | Detail |
|-------|--------|
| Name | Charitopoulos Kyriazis |
| University | Aristotle University of Thessaloniki |
| Department | Mechanical Engineering |


---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*
