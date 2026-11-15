# Local input archive

`runs.zip` is the unchanged archive supplied by the user. Its size and SHA-256
are recorded in each output's `provenance.json`. This directory ignores archives
and model data so they are not accidentally added to Git.

The analysis opens the archive and reads selected CSV/JSON files and model JSON
metadata. It does not extract replay buffers, load a policy, unpickle objects,
or run a simulator. Small, original evaluation files are copied to each output's
`source_data/` directory for review and future CSV-only reproduction.
