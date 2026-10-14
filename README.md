# Federated Graph-Based Centrality Analysis for IoT Network Security

This repository contains the experimental code developed for the Master's thesis:

**Federated Graph-Based Centrality Analysis for IoT Network Security**  
**Melissa El Husseini - Master 2 Data Science, Lebanese University**  
Research internship conducted with **CESI LINEACT, Strasbourg, France**.


The project studies how graph partitioning in federated settings affects node centrality measures and how this distortion influences downstream intrusion-detection tasks.

---

## Research Overview

Network traffic is represented as a graph in which:

- **Nodes** represent IP addresses.
- **Edges** represent communications between IP addresses.
- **Edge weights** represent the number of network flows between two nodes.
- Nodes and edges are associated with attack/benign information derived from the traffic data.

The study evaluates how splitting the global graph across multiple federated clients changes centrality values and rankings.

The main centrality measures used in the final stages are:

- Degree
- Betweenness
- Closeness
- PageRank

The main evaluation measures include:

- Spearman rank correlation
- Pearson correlation
- Edge preservation
- Attack-edge preservation
- Downstream classification performance
- Top-K threat-node coverage

---

## Main Research Questions

The experiments investigate:

1. How much graph partitioning distorts centrality measures.
2. Which partitioning strategies preserve graph structure best.
3. How community-detection algorithms affect federated graph quality.
4. Why centrality distortion occurs.
5. Whether distorted centralities can be corrected.
6. Whether centrality distortion affects downstream intrusion detection.
7. Whether a distortion-aware partitioner can reduce structural loss.
8. Whether the findings transfer to a graph outside the network-security domain.

---

## Datasets

The repository is designed around the following datasets:

### Network-security datasets

- **NF-ToN-IoT**
- **CIC-ToN-IoT**
- **IoT-23**

### Cross-domain dataset

- **Amazon Computers**

The raw datasets are **not included in this repository** because of their size.

Place the required dataset files in the repository root or update the dataset path/configuration inside the corresponding script.

For the main NF-ToN-IoT experiments, the expected file name is:

```text
NF-ToN-IoT.csv
```

---

## Repository Structure

```text

Centrality_Cen_Fed_Thesis/│
├── phase_01_baseline/
│   └── baseline_mae.py
│
├── phase_02&04_partitioning/
│   └── partitioning_strategy_comparison.py
│
├── phase_03&04_community_detection/
│   └── community_algorithm_comparison.py
│
├── phase_05_distortion/
│   └── distortion_study.py
│
├── phase_06_correction/
│   ├── correction_before_after.py
│   ├── correction_comparison.py
│   ├── step_a_build_features.py
│   └── step_b_train_test.py
│
├── phase_07_downstream_impact/
│   ├── experiment_1_flow_classification/
│   ├── experiment_2_topk/
│   ├── experiment_3_sgc/
│   ├── shared/
│   ├── supplementary_models/
│   └── support/
│
├── phase_08_dap/
│   └── distortion_aware_partitioner.py
│
└── phase_09_cross_domain/
    ├── build_amazon_global.py
    ├── build_amazon_local.py
    └── correct_and_topk_amazon.py
```

---

# Experimental Phases

## Phase 1 — Baseline Centrality Distortion

**Folder:** `phase_01_baseline`

The first experiment establishes the baseline by comparing global and federated centrality values using Mean Absolute Error (MAE).

Partitioning strategies include:

- Random
- Degree-balanced
- Stratified
- Community-based

Run:

```bash
python phase_01_baseline/baseline_mae.py
```

---

## Phases 2 and 4 — Partitioning Strategy Analysis

**Folder:** `phase_02&04_partitioning`

This phase compares the four main partitioning strategies across different numbers of federated clients.

Main strategies:

- Random
- Degree-balanced
- Stratified
- Community-based

Typical client counts:

```text
K = 3, 5, 7
```

The analysis focuses on ranking preservation and structural preservation.

Run:

```bash
python "phase_02&04_partitioning/partitioning_strategy_comparison.py"
```

The same analysis framework can also be reused with the other network datasets by adjusting the dataset configuration and column mappings.

---

## Phases 3 and 4 — Community Detection Analysis

**Folder:** `phase_03&04_community_detection`

This phase compares community-detection methods used to create federated graph partitions.

The evaluated configurations include:

- Greedy Modularity
- Girvan-Newman
- Label Propagation
- Infomap
- Louvain
- Leiden

Different Louvain and Leiden resolution settings are also evaluated.

Run:

```bash
python "phase_03&04_community_detection/community_algorithm_comparison.py"
```

Some configurations require:

```bash
pip install python-igraph leidenalg
```

---

## Phase 5 — Centrality Distortion Analysis

**Folder:** `phase_05_distortion`

This phase studies the relationship between graph cuts and centrality distortion.

The analysis evaluates whether nodes that lose more cross-client connections experience larger changes in their local centrality rankings.

Run:

```bash
python phase_05_distortion/distortion_study.py
```

This phase supports the identification of two main distortion mechanisms:

1. Loss of cross-client edges.
2. Changes caused by computing centrality on smaller local graphs.

---

## Phase 6 — Centrality Correction

**Folder:** `phase_06_correction`

This folder contains the correction-development experiments used during the study.

Included scripts:

### `correction_before_after.py`

Evaluates centrality values before and after applying correction.

### `correction_comparison.py`

Compares alternative correction approaches.

### `step_a_build_features.py`

Builds node-level features required for learned correction models using different client partitions.

### `step_b_train_test.py`

Trains and evaluates learned correction models across different federated partitions.

The correction experiments include:

- Exact degree correction
- Analytical correction approaches
- Learned correction models for centralities that depend on global graph structure

Run individual scripts from the repository root, for example:

```bash
python phase_06_correction/step_a_build_features.py
python phase_06_correction/step_b_train_test.py
```

---

## Phase 7 — Downstream Impact

**Folder:** `phase_07_downstream_impact`

This phase evaluates whether centrality distortion has a practical effect on intrusion-detection tasks.

It contains three main experiments.

### Experiment 1 — Flow Classification

**Folder:**

```text
phase_07_downstream_impact/experiment_1_flow_classification/
```

The experiment compares flow-classification performance under different centrality conditions:

- Global centralities
- Local centralities
- Corrected centralities
- No centrality features

Supporting scripts build global, local, and corrected flow-level feature tables.

---

### Experiment 2 — Top-K Threat-Node Identification

**Folder:**

```text
phase_07_downstream_impact/experiment_2_topk/
```

This experiment ranks nodes by centrality and evaluates whether the most important nodes selected from local or corrected rankings match the global ranking.

It also measures **attack-flow coverage** when only the Top-K nodes are monitored.

Run:

```bash
python phase_07_downstream_impact/experiment_2_topk/topk_attack_coverage.py NF-ToN-IoT.csv phase7_centralities.csv
```

Typical values of K are:

```text
5, 10, 20, 50
```

---

### Experiment 3 — Graph-Aware Node Classification with SGC

**Folder:**

```text
phase_07_downstream_impact/experiment_3_sgc/
```

Simple Graph Convolution (SGC) is used to test how centrality quality affects graph-aware node classification.

The main conditions are:

- Global centralities
- Local centralities
- Corrected centralities
- Zero centrality

The final experiment uses **one-hop propagation**.

Run:

```bash
python phase_07_downstream_impact/experiment_3_sgc/sgc_node_classification.py NF-ToN-IoT.csv phase7_centralities.csv --hops 1
```

---

### Supplementary Models

The `supplementary_models` directory contains additional model experiments used to study whether the impact of centrality distortion changes across model families.

These include conventional machine-learning/deep-learning models and graph neural-network variants.

---

## Phase 8 — Distortion-Aware Partitioning (DAP)

**Folder:** `phase_08_dap`

DAP is a graph partitioning method designed specifically to reduce the structural losses responsible for centrality distortion.

DAP starts from a Louvain partition and repeatedly evaluates boundary-node moves.

Its objective combines structural loss with a client-balance penalty:

```text
J(P) =
    structural loss
    + λ × client imbalance
```

A node move is accepted only when it decreases the objective.

Run:

```bash
python phase_08_dap/distortion_aware_partitioner.py NF-ToN-IoT.csv --clients 3
```

Different client counts can be tested using:

```bash
--clients 3
--clients 5
--clients 7
```

The main outputs include:

- Client assignments
- DAP optimization history
- Edge preservation
- Attack-edge preservation

---

## Phase 9 — Cross-Domain Validation

**Folder:** `phase_09_cross_domain`

The final phase evaluates whether the centrality-distortion and correction findings transfer beyond network-security graphs.

The **Amazon Computers** co-purchase graph is used as a non-network-security graph.

Scripts:

```text
build_amazon_global.py
build_amazon_local.py
correct_and_topk_amazon.py
```

The workflow:

1. Builds the global Amazon graph and centralities.
2. Builds federated/local graph centralities.
3. Applies correction.
4. Compares the recovered Top-K rankings with the global graph.

---

# Environment Setup

Python 3 is required.

A virtual environment is recommended.

### Windows

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install the main dependencies:

```bash
pip install -r requirements.txt
```

For the full community-detection comparison:

```bash
pip install python-igraph leidenalg
```

Some supplementary deep-learning experiments may require additional packages depending on the selected model.

---

# Running the Experiments

Most scripts are designed to be run from the repository root.

Example:

```bash
cd Centrality_Cen_Fed_Thesis
.venv\Scripts\activate
python phase_01_baseline/baseline_mae.py
```

For scripts that accept the dataset as a command-line argument:

```bash
python phase_08_dap/distortion_aware_partitioner.py NF-ToN-IoT.csv --clients 3
```

Generated outputs are normally written to phase-specific output directories or CSV files.

---

# Reproducibility Notes

The repository contains code from different stages of the research process.

Some early experiments use a larger set of centrality measures, while the later thesis analysis standardizes the main evaluation around:

```text
Degree
Betweenness
Closeness
PageRank
```

The repository preserves these experimental stages rather than rewriting all scripts into one single implementation because the evolution of the methodology is part of the research process.

Small numerical differences may occur across Python, NetworkX, NumPy, scikit-learn, and community-detection library versions.

Randomized procedures use fixed seeds where applicable to improve reproducibility.

---

# Main Findings Represented by the Code

The experimental pipeline supports the main findings of the thesis:

- Graph partitioning can substantially distort local centrality values and rankings.
- Partitioning strategies that ignore graph structure lose a large fraction of edges as the number of clients increases.
- Community-aware partitioning preserves substantially more graph structure.
- Centrality distortion is strongly connected to the loss of cross-client edges and changes in local graph size.
- Degree centrality can be corrected analytically using local structural information.
- Learned correction methods can improve centralities whose values depend on broader graph structure.
- The practical effect of distortion depends on how centrality is used downstream.
- Distortion is particularly important when centrality determines a threat ranking or is propagated through a graph-aware model.
- DAP improves the graph partition itself by explicitly reducing structural loss while considering client balance.
- Some correction results also transfer to a graph outside the network-security domain.

---

# Data and Code Organization

Large raw datasets, generated caches, virtual environments, and experiment outputs should not be committed to the repository.

Recommended `.gitignore` entries include:

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
*.pkl
*.pickle
output*/
outputs*/
results*/
```

Dataset files can also be excluded if they are stored locally:

```gitignore
NF-ToN-IoT.csv
CIC-ToN-IoT.csv
IoT-23.csv
```

---

# Author

**Melissa El Husseini**  
Master 2 Data Science  
Lebanese University

Master's thesis research conducted in collaboration with **CESI LINEACT, Strasbourg, France**.

---

## License

This repository is provided for academic and research purposes.

Please cite the thesis or repository if this code is reused in academic work.
