# 🏭 Polystyrene Batch-Plant Production Scheduling (STN / MILP)

A Mixed-Integer Linear Programming (MILP) model that finds the **profit-maximizing 24-hour production schedule** for a multi-stage polystyrene batch plant, using the **State-Task Network (STN)** formulation — solved in Python with **GAMSPy** and visualized with Gantt charts and inventory profiles.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Production Planning & Control
**Instructor:** Georgios Georgiadis, Assistant Professor
**Author:** Kyriazis Charitopoulos · 
**Date:** May 2026

---

## ⚙️ Tech Stack

- Python 3
- [GAMSPy](https://gamspy.readthedocs.io/) (GAMS modelling in Python)
- pandas, NumPy
- Matplotlib (Gantt charts & inventory profiles)
- MILP solver (via GAMS)

---

## 🧪 The Problem

A polystyrene plant converts raw **Feed** into final products through five batch stages. The process is modelled as a **State-Task Network**: *tasks* (processes) consume and produce *states* (materials) on shared *units* (equipment).

```
Feed → Premixing → Reaction → Screening → ⟨BeadSize1..4⟩
                                            ├─ BeadSize3, BeadSize4  → final products
                                            ├─ BeadSize1 → Sparging1 → Blending1 → Prod1
                                            └─ BeadSize2 → Sparging2 → Blending2 → Prod2
```

**Stages & timing:** Premixing (2 hr) · Reaction (6 hr, 2.5% evaporation loss) · Screening (semi-continuous, 12.7 tn/hr, splits 20/44/26/10%) · Sparging (4 hr) · Blending (1 hr).

**Equipment (21 units):** 2 premixers, 4 reactors (2×2 in parallel), 1 screener, 5 Sparger1 + 7 Sparger2 units, 2 blenders, and 1 shared 20 tn silo.

**Twist:** Several intermediates have **no external storage** (Premix, BeadSize1, BeadSize2), so they must be *held inside the equipment* via dedicated **Hold tasks**, and the shared silo can hold **only one** of BeadSize1 / BeadSize2 at a time.

---

## 🧮 The Model

**Indices** — `i` tasks · `s` states (materials) · `j` units (equipment) · `t` time periods (1…25, with δ = 1 hr, H = 24).

**Decision variables**
- `W[i,j,t]` — **binary**: 1 if task *i* starts on unit *j* at period *t*
- `B[i,j,t]` — batch size (tn) of task *i* on unit *j* at *t* (continuous ≥ 0)
- `S[s,t]` — inventory (tn) of material *s* at *t* (continuous ≥ 0)

**Objective** — maximise end-of-horizon revenue:

```
max profit = Σ_{s ∈ {BeadSize3, BeadSize4, Prod1, Prod2}}  price[s] · S[s, H+1]
```
(prices: BeadSize3 = 0.4, BeadSize4 = 0.3, Prod1 = 2.0, Prod2 = 2.1 k€/tn)

**Constraints**
- **Unit assignment** — each unit runs at most one task at a time (occupied for the task's full duration)
- **Capacity** — `Vmin·W ≤ B ≤ Vmax·W` (linking start to batch size)
- **Mass balance** — inventory carries over + produced − consumed
- **Storage** — `S ≤ C[s]`; no-intermediate-storage states forced to 0
- **Hold interconnection** — can't hold more than the available inventory; Hold only after a finished task or a prior Hold
- **Silo mutual exclusion** — `W_HoldBS1,Silo,t + W_HoldBS2,Silo,t ≤ 1`

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `STN_cl_EN.py` | Full GAMSPy STN model + Gantt / inventory plotting |
| `report.pdf` | Report: STN diagram, mathematical model, results, conclusions |
| `docs/assignment.pdf` | Original assignment brief (reference) |
| `gantt_chart.png` | Generated production schedule (Gantt) |
| `state_profiles.png` | Generated inventory profiles |

---

## 🚀 Running

**Requirements:** Python 3.8+, a working GAMS/GAMSPy installation

```bash
pip install gamspy pandas numpy matplotlib
python STN_cl_EN.py
```

> ℹ️ GAMSPy requires a GAMS backend (a free community/demo license covers a model of this size).
> The script writes `gantt_chart.png` and `state_profiles.png` — see the note below on output paths.

---

## 📊 Results

**Maximum total profit: 193.696 k€** over the 24-hour horizon.

Key findings:
- **Screening is the bottleneck.** With a single semi-continuous screener (12.7 tn/hr) and four reactors feeding it large batches (20 & 40 tn), `ReactorProduct` accumulates heavily — the screener runs flat-out from t = 10 to t = 24.
- **Spargers & blenders run continuously** with variable batch sizes (0.5–5.0 tn at spargers, 0.5–8.0 tn at blenders) to push material toward the high-value Prod1 / Prod2 without overflowing intermediate storage.
- **Hold tasks and the silo behave correctly:** materials with no external storage are held inside equipment, and the silo alternates between BeadSize1 and BeadSize2 — never both at once.

---

## 👥 Author

| Name |
|------|
| Kyriazis Charitopoulos |

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*

