# Sparse graph and spectral decomposition

## Graph contract

Each segment is one node. Node order is deterministic by trajectory ID and closed start step. Segment embeddings are L2-normalized for cosine calculations, while persisted Mid centroids use the arithmetic mean of the original segment embeddings.

The semantic and transition neighbor counts follow the fixed rule:

```text
k = clip(ceil(log2(segment_count)), 10, 30)
```

For sets smaller than `k + 1`, k is clamped to `segment_count - 1`. Exact cosine neighbors come from scikit-learn's brute-force `NearestNeighbors`; the candidate mask is the symmetric union of semantic and transition neighbors.

Outcome smoothing uses semantic neighbors and the fixed own-outcome prior:

```text
rho_u = (score_u + sum(S_uv * score_v)) / (1 + sum(S_uv))
```

Scores and rho are clipped to `[0, 1]`. Candidate-edge components are:

```text
S_uv = max(0, cosine(segment_u, segment_v))
T_uv = max(0, cosine([previous_u, next_u], [previous_v, next_v]))
O_uv = clip(1 - abs(rho_u - rho_v), 0, 1)
```

The base weight is the equal mean of enabled components. No Transition and No Outcome remove only their named base component. No Contrastive retains S/T/O and replaces signed adjacency with `A = B`.

Full signed adjacency and normalized Laplacian follow the experiment specification. Degree uses `sum(abs(A_uv))`; zero-degree nodes use inverse square root zero, leaving a finite diagonal-one Laplacian row. Explicit sparse zero entries are removed before persistence.

## Spectral contract

The allowed eigengap range is frozen to `[2, 20]`, clamped by node count and available non-degenerate eigenvectors. Eigenvectors with non-finite values or near-zero row variance are excluded. Equal eigengaps choose the smaller K.

Sparse decomposition uses SciPy `eigsh` with a seeded initial vector. Small matrices use deterministic dense `eigh`. The selected rows are normalized before scikit-learn KMeans with `random_state=42`, `n_init=20`, and `max_iter=300`. Whole-column eigenvector sign flips preserve row distances; a focused test verifies the resulting canonical partition is unchanged.

Cluster labels are canonicalized by the lexicographically smallest member segment ID. Mid IDs therefore remain stable as `mid_0000`, `mid_0001`, and so on when the partition is unchanged. Semantic-Only directly clusters the same segment embeddings with the Full build's K and random state.

## Pilot results

Full Trace2Tower on the Flash pilot produced:

| Benchmark | Segments | kNN | Undirected edges | K | Positive entries | Negative entries |
|---|---:|---:|---:|---:|---:|---:|
| ALFWorld | 34 | 10 | 309 | 6 | 284 | 334 |
| WebShop | 26 | 10 | 234 | 3 | 468 | 0 |

The WebShop pilot contains four full successes and one partial-score completion, so smoothed rho lies in `[0.8253, 1]` and signed adjacency is positive. Synthetic success/failure blocks separately prove negative adjacency behavior.

Pilot cluster counts by variant:

| Benchmark | Full | No Transition | No Outcome | No Contrastive | Semantic-Only |
|---|---:|---:|---:|---:|---:|
| ALFWorld | 6 | 6 | 4 | 2 | 6 |
| WebShop | 3 | 3 | 3 | 3 | 3 |

Semantic-Only reused Full K exactly. Rebuilding ALFWorld Full with identical relative arguments produced byte-identical sparse matrices, spectral arrays, clusters, and report.
