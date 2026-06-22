# 🏭 Optimal Design of a Production & Distribution Network

A Mixed-Integer Linear Programming (MILP) model that decides **where to open a distribution center in Germany** to serve a manufacturer's agents at minimum total cost — solved in Python with **GAMSPy**, in both a deterministic and a stochastic (multi-scenario) formulation.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Production Planning & Control
**Instructor:** Georgios Georgiadis, Assistant Professor
**Author:** Kyriazis Charitopoulos · AEM 7137
**Date:** May 2026

---

## ⚙️ Tech Stack

- Python 3
- [GAMSPy](https://gamspy.readthedocs.io/) (GAMS modelling in Python)
- pandas, NumPy
- MILP solver (via GAMS)

---

## 📦 The Problem

A manufacturer of household appliances runs **two factories in Greece** (Nea Santa & Komotini) and wants to open **one distribution center (DC)** in Germany to serve **10 customer/agent zones**. There are **8 candidate DC locations** and **12 products**.

The goal is to choose the single DC location that **minimises total network cost**, where total cost is the sum of:

- 🏗️ DC **installation** cost (fixed, per chosen location)
- 🔧 **production** cost (per product, per factory)
- 🚚 **transport** cost (factory → DC, and DC → customer)
- 📦 **handling** cost (per unit, at the DC)

subject to: meeting all customer demand, a single chosen DC, DC storage capacity (50,000 m²), and per-link flow limits (90,000 units).

---

## 🧮 The Model

**Indices** — `i` products · `j` factories · `k` candidate DCs · `l` customer zones (and `s` demand scenarios in Part C).

**Decision variables**
- `x[i,j,k]` — units of product *i* produced at factory *j* and shipped to DC *k* (continuous ≥ 0)
- `w[i,k,l]` — units of product *i* shipped from DC *k* to customer *l* (continuous ≥ 0)
- `y[k]` — **binary**: 1 if DC *k* is opened, 0 otherwise

**Objective** — minimise total cost:

```
min Z = Σ c·x   (production)
      + Σ b·y   (installation)
      + Σ h·w   (handling)
      + Σ cd·x  (transport factory → DC)
      + Σ cc·w  (transport DC → customer)
```

**Constraints**
- **Demand:** every customer's demand for every product is met
- **Flow balance:** what enters a DC leaves it
- **Single DC:** Σ y[k] = 1
- **Capacity:** Σ v·w ≤ 50,000 · y[k] per DC
- **Transport limits:** ≤ 90,000 · y[k] on each link (big-M style — non-binding here)

---

## 🏗️ Approach

**Part A & B — Deterministic model.** Solve the MILP on the base demand scenario to find the optimal DC, the minimum total cost, and the production plan (which product is made at which factory, in what quantity).

**Part C — Stochastic model.** Build 5 demand scenarios around the base forecast and minimise **expected** total cost:

| Scenario | Demand factor | Probability |
|----------|---------------|-------------|
| lower | ×0.75 | 5% |
| low | ×0.90 | 15% |
| basic | ×1.00 | 60% |
| high | ×1.10 | 15% |
| higher | ×1.25 | 5% |

The model is solved for each candidate DC (fixing `y[k]=1` in turn) and the location minimising the probability-weighted expected cost is selected.

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `Project_1_Charitopoulos_7137_EN.py` | Full GAMSPy model — deterministic + stochastic |
| `report.pdf` | Report: mathematical model, methodology, results, interpretation |
| `docs/assignment.pdf` | Original assignment brief (reference) |

---

## 🚀 Running

**Requirements:** Python 3.8+, a working GAMS/GAMSPy installation

```bash
pip install gamspy pandas numpy
python Project_1_Charitopoulos_7137_EN.py
```

> ℹ️ GAMSPy requires a GAMS backend (a free community/demo license covers a model of this size).

---

## 📊 Results

**Optimal DC location: Erfurt** — in both the deterministic and the stochastic formulation.

| Formulation | Total Cost |
|-------------|-----------|
| Deterministic (base demand) | **€4,412,384.92** |
| Stochastic (expected cost) | **€4,414,280.32** |

**Cost breakdown (deterministic):** production dominates (~€2.99M), followed by factory→DC transport (~€1.12M); installation (€96,605) and handling (~€48.5k) are smaller.

**Production plan:** 10 of the 12 products are cheapest to make at **Nea Santa**; only **M1** and **JK1** are produced at **Komotini** (where their unit cost is 3× and 5× lower, respectively).

**Second-best location:** Munich — the marginal value on the binary tells us the next-cheapest choice would add ≈ €96,605.

---

## 👥 Author

| Name | Student ID |
|------|------------|
| Kyriazis Charitopoulos | 7137 |

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*
