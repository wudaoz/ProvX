# NoDoze anomaly detection (baseline)

This directory contains a lightweight implementation of **NoDoze** (NDSS 2019), used as a baseline for provenance triage/anomaly scoring.

Paper: *"NODOZE: Combatting Threat Alert Fatigue with Automated Provenance Triage"* (NDSS 2019).

## Core scoring

1. **Transition probability** \(M_\epsilon\)

```
M_epsilon = |Freq(epsilon)| / |Freq_src_rel(epsilon)|
```

where `epsilon = (src_type, rel, dst_type)` is an edge-type triple.

2. **Regular score** (RS)

```
RS = IN(src) × M_epsilon × OUT(dst)
```

3. **Anomaly score** (AS)

```
AS = 1 - RS
```

Higher anomaly score indicates a more suspicious edge.

## Files

```
models/
├── nodoze.py               # NoDoze core implementation
├── nodoze_data_adapter.py  # Adapters for dataset formats
├── nodoze_utils.py         # Helper utilities
└── README_NoDoze.md        # This document
```

## Usage (examples)

The exact CLI depends on how you integrated NoDoze into your experiment scripts, but typical workflows are:

- **Build a frequency dictionary** from background/training data
- **Score edges** in a target graph/subgraph
- **Return Top-K** most anomalous edges

If you need a template weights table when background data is missing, see utilities in `models/nodoze_utils.py`.

## Notes on data fields

NoDoze generally requires:

- **Node types** (e.g., process, file, socket, ...)
- **Edge relation types** (e.g., read/write/execute/connect, ...)

If your stored graphs do not contain those fields, you may need to reconstruct them in preprocessing or provide a fallback weight table.

