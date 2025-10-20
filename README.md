# QSVM-Iris
# Quantum Iris Binary Classifier – Variational Circuits A & B

This project implements and compares **two quantum circuit classifiers** (Proposal A and Proposal B) for a **binary classification** task on the Iris dataset using **PennyLane**. It also estimates each circuit’s **expressibility** to reason about representational power vs. practical trade‑offs like depth and noise.

> TL;DR – You get: reproducible preprocessing → two distinct circuit families → clean vs. noisy training/eval → a simple expressibility proxy (variance + entropy) → a short, opinionated analysis of when to use which.

---

## What the code does (at a glance)

1. **Loads & preprocesses Iris** for a 2‑class problem (Setosa vs. Versicolor).
2. **Maps features → angles** for 2 qubits via `StandardScaler → PCA(2) → MinMaxScaler((0, π))`.
3. Builds **two different variational circuits** (A and B), each with distinct data encoding and entanglement patterns.
4. **Trains** both models with binary cross‑entropy using autograd (full‑batch, gradient descent).
5. **Evaluates** accuracy in **clean** and **noisy** simulation modes (noisy uses `default.mixed` and layer‑level depolarizing channels).
6. **Estimates expressibility** by sampling random parameters and inputs, then computing **output‑probability variance** and **histogram entropy**.
7. Prints a **summary “winner”** for clean accuracy, noisy accuracy, and expressibility.

---

## Dataset & Preprocessing

- **Subset:** Iris Setosa (label 0) vs. Iris Versicolor (label 1). Virginica is excluded.
- **Pipeline:**  
  `StandardScaler` → `PCA(n_components=2)` → `MinMaxScaler(feature_range=(0, π))`  
  Rationale:
  - **Standardization** centers and scales features to stabilize optimization.
  - **PCA to 2D** matches the **2‑qubit** budget while retaining maximal variance in a linear subspace.
  - **MinMax to [0, π]** produces **angle-encoded** features that are well‑conditioned for **RX/RY** rotations and avoid wrapping issues beyond 2π.
- **Train/test split:** stratified, `test_size=0.2`, fixed `SEED` for reproducibility.

---

## Models

Both models output an expectation ⟨Z₀⟩ which is mapped to a probability via `p = 0.5 * (⟨Z⟩ + 1)`. Loss is binary cross‑entropy.

### Proposal A (shallower entangling block)

- **Layers:** `A_LAYERS = 2`
- **Per‑layer parameters:** `6` → total trainables `2 × 6 = 12`
- **Feature map:** `RX(x0) ⊗ RX(x1)` (angle encoding)
- **Trainable local rotations:** `RY` and `RZ` on each qubit
- **Entanglement:** `CZ(0,1)`
- **Extra bias-like twist:** final `RZ(ent_theta + bias)` on wire 0
- **Device:** `default.qubit` (clean) or `default.mixed` (noisy)
- **Readout:** `expval(PauliZ(0))`

**Design intent:** small depth, strong single‑qubit expressivity + one controlled‑phase interaction. Good baseline with **low noise surface area** and **fast simulation**.

---

### Proposal B (deeper with shifted re-encoding)

- **Layers:** `B_LAYERS = 3`
- **Per‑layer parameters:** `4` → total trainables `3 × 4 = 12` (+ fixed **per‑layer shift** parameters)
- **Feature map:** `RY(x0) ⊗ RY(x1)`
- **Shifted re-encoding:** each layer ends with `RY(x0 + shift_l)` and `RY(x1 + shift_l)` (shifts are fixed and linearly spaced in `[0.1, 0.9]`).
- **Entanglement:** layer block includes a two‑qubit interaction (see `layer_block_B` in code).
- **Noise-ready:** when `noise=True`, the node uses `default.mixed`; the layer block injects **depolarizing noise** (with probs `P1`/`P2`) at designated points.
- **Readout:** `expval(PauliZ(0))`

**Design intent:** slightly **deeper** architecture with **periodically re‑encoded features** (via shifts) to break simple symmetries and encourage richer decision boundaries. Good for **expressivity**; potentially more sensitive to noise.

> Note: Exact two‑qubit gate choice (CNOT/CZ/controlled‑rotations) and where depolarizing channels sit are defined inside `layer_block_B` and are easy to tweak.

---

## Training

- **Optimizer:** simple gradient descent (autograd/backprop), learning rate `LR = 0.12`, `EPOCHS = 60`
- **Loss:** Binary cross‑entropy on model probability `p`
- **Batching:** full‑batch over the training split
- **Noise toggles:** Train/eval in both **clean** and **noisy** modes

**Why BCE on ⟨Z⟩→p?** It’s stable and directly aligned with probabilistic readout. Mapping ⟨Z⟩ to `[0,1]` keeps gradients informative even when expectations saturate near ±1.

---

## Expressibility Proxy

The script estimates expressibility using two summary statistics of the model’s **output distribution** (with **random parameters** and **random inputs**):

- **Variance of probabilities** across samples
- **Histogram entropy** (20 bins over `[0,1]`)

> Higher **variance** + **entropy** typically indicates the circuit can realize a **wider spread** of functions (i.e., richer representational capacity). This is a **proxy**, not a formal measure; it’s useful for **relative** comparisons between proposals A and B.

Config: `EXP_K=64` random parameter draws × `EXP_M=32` inputs per draw (tunable).

---

## Evaluation & What to Expect

The script prints (a) **clean accuracy**, (b) **noisy accuracy** (with depolarizing noise `P1`, `P2`), and (c) **expressibility metrics** (variance & entropy), followed by simple “winners.”

**Qualitative expectations:**

- On Setosa vs. Versicolor (close to linearly separable), **both** circuits should achieve **high clean accuracy** once trained.
- **Proposal A** (shallower) often shows **slightly better noise robustness**, fewer train‑time quirks, and faster iteration.
- **Proposal B** can exhibit **higher expressibility** due to depth + re‑encoding shifts, which may help if the boundary is more nonlinear—but can be **more sensitive to noise** and hyperparameters.
- Because both have **~12 trainables**, the main differences come from **architecture** (feature map, entanglement placement, depth, and shifts) rather than parameter count.

> Actual numbers depend on the RNG seed, split, and hyperparameters. Use the printed output to decide the “winner” under your constraints.

---

## Practical Considerations

- **Scalability:** With 2 qubits / 2 features, everything is fast. For higher‑dimensional data, extend qubits or use **feature maps (e.g., tensor product features, embedding blocks)** and consider **sparse entanglement** for depth control.
- **Circuit depth vs. noise:** Depth increases expressivity but also **noise surface area**. Use **error‑mitigation** or **hardware‑aware compilation** when targeting real devices.
- **Preprocessing:** Dimensionality reduction (PCA) is a **strong lever**; pick `n_components` to match qubit budget. Alternatives: feature selection, learned embeddings, or kernel methods.
- **Shots vs. analytic:** Here, training is analytic (`shots=None`). On hardware, expect **shot noise**; set a reasonable shots budget and consider **stochastic training**.

---

## How to run

### In a notebook
- Open `QSVMIRIS.ipynb` and run all cells.

### As a script (optional)
The notebook contains a `main()` and a `if __name__ == "__main__": main()` guard. If you export to a Python script (e.g., `jupyter nbconvert --to script QSVMIRIS.ipynb`) you can run:

```bash
python QSVMIRIS.py
```

### Requirements
- Python 3.10+ (recommended)
- `pennylane`, `scikit-learn`, `numpy`

Install:
```bash
pip install pennylane scikit-learn numpy
```

---

## Answers to the assignment prompts

**Q1. Create at least two different circuit-based proposals, each with distinct architectures and layers.**  
**A.** Proposal **A** and **B** meet this: different feature maps (RX vs. RY), different **layer counts** (2 vs. 3), and different layer structures (CZ + bias vs. shifted re‑encoding with a different entangling block).

**Q2. Clearly describe the design choices for each proposal.**  
**A.** See “Models” above: per‑layer gates, entanglement structure, parameter counts, and readout are detailed.

**Q3. Use the Iris dataset restricted to a binary classification task.**  
**A.** Yes—Setosa vs. Versicolor (`y ∈ {0,1}`), with stratified split.

**Q4. Apply appropriate classical preprocessing.**  
**A.** StandardScaler → PCA(2) → MinMax([0, π]) for stable optimization, 2‑qubit matching, and angle‑friendly ranges.

**Q5. Explain the rationale behind the chosen preprocessing methods.**  
**A.** Standardization stabilizes learning; PCA matches qubit budget while preserving variance; MinMax to `[0, π]` harmonizes with rotation encodings and avoids periodicity artifacts.

**Q6. Evaluate the expressibility of each circuit to identify representational limits.**  
**A.** A custom **expressibility proxy** computes **probability variance** and **histogram entropy** over random parameter draws and inputs. Higher values indicate richer representational capacity.

**Q7. Discuss how the architecture affects the ability to represent complex boundaries.**  
**A.** Depth, entanglement placement, and re‑encoding shifts (Proposal B) generally **increase expressivity**, enabling more complex, nonlinear decision boundaries—but they **amplify noise sensitivity**. Proposal A is **shallower** and may underfit highly nonlinear boundaries but provides stronger **noise resilience** and **training stability**.

**Q8. Compare performance and analyze strengths/weaknesses.**  
**A.** Empirically (see your run’s printout):  
- **A**: Fast, simple, robust under noise; may be less expressive.  
- **B**: More expressive (often higher proxy scores), possibly higher clean accuracy on harder boundaries; more sensitive to noise/hyperparams.

**Q9. Provide reasoning based on expressibility, accuracy, and practical considerations.**  
**A.** If your hardware/noise budget is tight or latency matters, **A** is preferable. If you need more expressive decision surfaces and can tolerate/mitigate noise, **B** is attractive. Use the **expressibility proxy** and **noisy accuracy** to guide the trade‑off.

---

## Tuning tips

- Increase `A_LAYERS`/`B_LAYERS` to explore depth vs. noise.
- Try different entanglers (e.g., `CNOT`, `CZ`, `CRX`) or add parametrized two‑qubit gates.
- Replace PCA with feature selection or try learned classical embeddings.
- Adjust `LR`, `EPOCHS`, and add early stopping for stability.
- When simulating hardware, switch to finite **shots** and tune shot counts + depolarizing probabilities.

---

## File structure

- `QSVMIRIS.ipynb` – the main notebook with both proposals, training, evaluation, and expressibility logic.
- (Optional) `QSVMIRIS.py` – if you export the notebook to a script for CLI runs.

---

## License

Use, modify, and extend freely for academic or personal projects. If you publish, a citation/acknowledgment would be lovely :)

