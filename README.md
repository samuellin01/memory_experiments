# memory_experiments

A repository for running and analyzing memory compaction experiments across various software projects.

Experiment results live under the `sbp/` directory. Each experiment instance is stored in a subdirectory named `instance_<project>-<commit_hash>[-v<variant_hash>]`. Inside each instance there are one or more variant subdirectories (e.g. `no_compression`, `smart_context_30000_12000`, `smart_context_50000_20000`), each containing files such as `agent_output.log`, `metadata.json`, `patch.diff`, `token_usage.json`, and `traj_*.json`.

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

You can also point either command at a different directory with `--experiment-dir`:

```bash
python repo_manager.py list-instances --experiment-dir /path/to/other/sbp
python repo_manager.py file-counts --experiment-dir /path/to/other/sbp
```

### `visualize_token_usage.py` — Token usage visualization

Generates graphs and an HTML report comparing token usage across experiment variants.

```bash
python visualize_token_usage.py
```

Output is saved to the `output_graphs/` directory.
