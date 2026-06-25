# Demeter: Exploiting Wide-Area Resource Elasticity With Fine-Grained Orchestration

<!-- ![TON 2025](https://img.shields.io/badge/TON-2025-blue)
![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)
![Python](https://img.shields.io/badge/Python-3.9%2B-green) -->

This repository contains the source code for the paper:

> Exploiting Wide-Area Resource Elasticity With Fine-Grained Orchestration for Serverless Analytics, IEEE/ACM Transactions on Networking, 2025.

## Overview

Demeter is a fine-grained function orchestrator for geo-distributed serverless analytics. It targets analytics jobs represented as Directed Acyclic Graphs (DAGs), where each stage contains many short-lived serverless functions that may run near different data centers.

This version implements the journal-version design for exploiting wide-area resource elasticity:

- fine-grained per-function placement and multi-resource allocation;
- history-guided allocation pruning for efficient resource selection;
- elastic Degree-of-Parallelism (DoP) tuning for function congestion control;
- JumpHash-based bucket partitioning for elastic repartitioning.

---

## Requirements

### Python Runtime

- Python 3.9+
- `numpy`
- `PyYAML`
- `networkx`
- `pytest` for tests
- `torch` for neural actor/critic and training scaffolding
- `tqdm` for training utilities
- `prometheus-api-client` and `kubernetes` for cluster-side integrations

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Build Dependencies

The `pheromone/` directory is the Pheromone serverless platform tree from *Following the Data, Not the Function: Rethinking Function Orchestration in Serverless Computing*. It provides data buckets, data-triggered function orchestration, and shared-memory data exchange.

Building it requires a Linux-like development environment with:

- CMake
- GNU Make
- `g++` with C++14 support
- Protobuf compiler and development libraries
- ZeroMQ and yaml-cpp dependencies used by Pheromone
- Kubernetes tooling for cluster deployment

---

## Run Simulation

Run the built-in synthetic geo-distributed analytics workload:

```bash
python -m MARL.cli simulate \
  --config configs/demeter.yml \
  --jobs 64 \
  --mode normal \
  --output results/simulation.csv
```

---

## MARL Training

The simulator can run with the lightweight heuristic policy without PyTorch. Neural components are provided in:

- `MARL/networks.py`
- `MARL/training.py`
- `MARL/features.py`

Install PyTorch before using the training scaffold:

```bash
pip install torch tqdm
```
---

## Reference

- [Following the Data, Not the Function: Rethinking Function Orchestration in Serverless Computing](https://arxiv.org/abs/2109.13492)
- [Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider](https://www.usenix.org/conference/atc20/presentation/shahrad)
- [Resource Allocation in Serverless Query Processing](https://arxiv.org/abs/2208.09519)
- [Astrea: Auto-serverless Analytics Towards Cost-efficient and QoS-aware Data Processing in the Cloud](https://ieeexplore.ieee.org/document/10171385)
