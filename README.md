# Adaptive Multi-Order Node Embeddings

This repository contains the code, models, and experiments for the paper: **"The Challenge of Multi-Order Embeddings"**
*Amro Asali, June 2025*

## 📌 Overview

Graph Neural Networks (GNNs) often struggle to generalize across tasks like node classification and link prediction due to conflicting structural requirements. This work systematically evaluates adaptive multi-order embedding architectures that overcome these limitations, including:

* **Jumping Knowledge Networks (JKNet)** with MaxPool and LSTM aggregation.
* **MixHop**: multi-hop neighborhood mixing.
* A **novel Hybrid-JKN model** combining GCN and MixHop under JK aggregation.

We experiment across three learning settings:

* **Single-task Learning**
* **Cross-task Transfer**
* **Multi-task Learning**

Datasets used:

* **Cora** (citation network)
* **Amazon Photo** (co-purchase graph)

---

## 🧪 Directory Structure

```
.
├── Experiments_amznphoto
│   ├── Cross Task
│   │   └── crstask_amznphoto.ipynb
│   ├── Link prediction
│   │   └── linkprd_amznphoto.ipynb
│   ├── Multi Task
│   │   └── mtl_amznphoto.ipynb
│   └── Node Classification
│       └── nodecl_citeseer.ipynb
│
├── Experiments_cora
│   ├── Cross Task
│   │   └── crstask_cora.ipynb
│   ├── Link prediction
│   │   └── linkprd_cora.ipynb
│   ├── Multi Task
│   │   └── mtl_cora.ipynb
│   └── Node Classification
│       └── nodecl_cora.ipynb
│
├── GNN_project.pdf  ← Final project report
└── README.md
```

---

## 📊 Results Summary

| **Model**  | **Node Classification (Cora)** | **Link Prediction (Cora AUC)** | **Multi-task Accuracy (Photo)** |
| ---------- | ------------------------------ | ------------------------------ | ------------------------------- |
| GCN        | 0.7630                         | 0.7944                         | 0.9124                          |
| JKNet-Max  | 0.7640                         | 0.7996                         | 0.9150                          |
| MixHop     | 0.7880                         | 0.7116                         | 0.8641                          |
| Hybrid-JKN | 0.7870                         | **0.8357**                     | **0.9183**                      |

Full results and analysis available in [`GNN_project.pdf`](./GNN_project.pdf).

---

## 🧐 Key Insights

* **MixHop** excels in node classification due to its multi-hop feature mixing.
* **Hybrid-JKN** shows strong generalization across tasks, especially for link prediction.
* **JKNet-Max** provides stable, high accuracy with minimal complexity.
* Cross-task evaluation confirms that embeddings optimized for one task often degrade in another.

---

## 🚀 Getting Started

### Requirements

* Python 3.8+
* PyTorch
* PyTorch Geometric
* Scikit-learn
* tqdm, matplotlib

### Run Example

To run the node classification task on Cora:

```bash
cd Experiments_cora/Node\ Classification
jupyter notebook nodecl_cora.ipynb
```

For multi-task learning on Amazon Photo:

```bash
cd Experiments_amznphoto/Multi\ Task
jupyter notebook mtl_amznphoto.ipynb
```

