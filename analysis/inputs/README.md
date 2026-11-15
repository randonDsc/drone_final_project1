# Original Analysis Input

We preserve the original evaluation archive as `runs.zip`. Its byte size and SHA-256 are recorded in the analytical provenance. Large archives, model weights, and replay buffers in this directory are ignored by Git.

Our analysis reads selected CSV/JSON files and model JSON metadata without loading serialized policy objects or replay buffers. Preserved small evaluation and configuration files accompany the published analytical outputs in `source_data/`. Numerical reproduction can use that smaller source bundle.
