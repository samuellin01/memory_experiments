# memory_experiments

A repository for running and analyzing memory compaction experiments across multiple benchmarks and domains, including SWE-bench Plus (SBP) and OSWorld.

## Repository Structure

```
.
├── .gitignore
├── README.md
├── repo_manager.py
├── requirements.txt
├── sbp/
│   ├── README.md
│   ├── eval_scores_no_compression.json
│   ├── eval_scores_smart_context_50000_20000.json
│   └── instance_<commit>[-v<variant_hash>]/
│       ├── no_compression/
│       ├── smart_context_30000_12000/
│       └── smart_context_50000_20000/
└── osworld/
    └── <uuid>/
        ├── no_compression/
        └── smart_context_50000_20000/
```

## Experiment Directories

### `sbp/` — SWE-bench Plus experiments

Contains results from SWE-bench Plus tasks across multiple open-source projects (e.g. ansible/ansible, element-hq/element-web, flipt-io/flipt, future-architect/vuls, gravitational/teleport, internetarchive/openlibrary, navidrome/navidrome, nodebb/nodebb, protonmail/webclients, qutebrowser/qutebrowser, tutao/tutanota).

**Instance naming:** `instance_<org>__<repo>-<commit_hash>[-v<variant_hash>]`

Each instance directory contains one or more variant subdirectories (e.g. `no_compression`, `smart_context_30000_12000`, `smart_context_50000_20000`). Each variant subdirectory contains files such as `agent_output.log`, `metadata.json`, `patch.diff`, `token_usage.json`, and `traj_*.json`.

The `sbp/` directory also contains aggregated evaluation score JSON files:
- `eval_scores_no_compression.json`
- `eval_scores_smart_context_50000_20000.json`

### `osworld/` — OSWorld experiments

Contains results from OSWorld tasks. Task directories are named by UUID (e.g. `09a37c51-e625-49f4-a514-20a773797a8a`). Each task directory contains variant subdirectories (e.g. `no_compression`, `smart_context_50000_20000`), with the same per-variant file structure as `sbp/`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Tools

### `repo_manager.py` — Repository management CLI

A command-line tool for inspecting experiment instances and their contents.

#### Usage

```
python repo_manager.py <command> [options]
```

Run with `--help` for a full description:

```bash
python repo_manager.py --help
python repo_manager.py list-instances --help
python repo_manager.py file-counts --help
```

#### Commands

##### `list-instances`

Lists the names of all experiment instance directories.

```bash
python repo_manager.py list-instances
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment-dir DIR` | `sbp` | Root directory containing instance subdirectories |

Example output:

```
instance_ansible__ansible-0ea40e09...-v30a923fb...
instance_flipt-io__flipt-f808b4dd...
instance_navidrome__navidrome-8e640bb8...
...
```

> **Note:** `list-instances` filters for subdirectories whose names begin with `instance_`. OSWorld task directories use UUID names and will not be matched by this filter when pointing at `osworld/`.

##### `file-counts`

For each instance, shows the number of files in each variant subdirectory.

```bash
python repo_manager.py file-counts
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment-dir DIR` | `sbp` | Root directory containing instance subdirectories |

Example output:

```
instance_ansible__ansible-0ea40e09...-v30a923fb...
  no_compression: 5 files
  smart_context_30000_12000: 5 files
  smart_context_50000_20000: 6 files
instance_flipt-io__flipt-f808b4dd...
  no_compression: 5 files
...
```

You can point either command at a different directory with `--experiment-dir`:

```bash
python repo_manager.py list-instances --experiment-dir /path/to/other/sbp
python repo_manager.py file-counts --experiment-dir /path/to/other/sbp
```

> **Note:** Both commands filter for subdirectories whose names begin with `instance_`. OSWorld task directories use UUID names and will not be matched by this filter, so neither command will return results when pointed at `osworld/`.
