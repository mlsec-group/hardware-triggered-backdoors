# %%

import json
from pathlib import Path
import itertools
import pandas as pd

# --------------------
# Configuration
# --------------------
DTYPES = ["bfloat16", "float16"]

RUNDIRS = [
    "2025-10-25T10:37:02.070783_regal-dove",
    "2025-10-29T13:25:52.233029_fuzzy-rhino",
    "2025-11-03T17:53:11.086550_thick-rhino",
]

PLATFORMS = ["a40", "a100", "a100-mig40", "rtx6000", "h100"]

INDICES = range(100)

BASE_DIR = Path("output/experiments/dtype-change")  # change if needed


# --------------------
# Helper functions
# --------------------
def load_label_dtype(json_path: Path):
    """Load label_dtype from a json file. Return None if file is missing or invalid."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("label_dtype")
    except Exception:
        return None


# --------------------
# Main traversal
# --------------------
rows = []

for dtype in DTYPES:
    dtype_dir = BASE_DIR / f"dtype-{dtype}"

    for rundir in RUNDIRS:
        rundir_dir = dtype_dir / rundir
        if not rundir_dir.exists():
            continue

        # All unordered platform pairs
        for platform_A, platform_B in itertools.combinations(PLATFORMS, 2):

            # Try both orderings
            possible_pair_dirs = [
                f"{platform_A}-{platform_B}",
                f"{platform_B}-{platform_A}",
            ]

            for pair_dir_name in possible_pair_dirs:
                pair_dir = rundir_dir / pair_dir_name
                if not pair_dir.exists():
                    continue

                for index in INDICES:
                    index_dir = pair_dir / f"Index-{index}"
                    if not index_dir.exists():
                        continue

                    json_A = index_dir / f"y_dtype_{platform_A}.json"
                    json_B = index_dir / f"y_dtype_{platform_B}.json"

                    label_A = load_label_dtype(json_A)
                    label_B = load_label_dtype(json_B)

                    # Skip if either label is missing
                    if label_A is None or label_B is None:
                        continue

                    rows.append(
                        {
                            "dtype": dtype,
                            "rundir": rundir,
                            "index": index,
                            "platform_A": platform_A,
                            "platform_B": platform_B,
                            "label_dtype_A": label_A,
                            "label_dtype_B": label_B,
                            "label_mismatch": label_A != label_B,
                        }
                    )

# --------------------
# Create DataFrame
# --------------------
df = pd.DataFrame(rows)

# If nothing was found, fail early
if df.empty:
    print("No valid data found.")
else:
    print("Raw dataframe shape:", df.shape)

# --------------------
# Optional: explode platforms for per-platform grouping
# --------------------
df_long = df.copy()
df_long["platform"] = df_long[["platform_A", "platform_B"]].values.tolist()
df_long = df_long.explode("platform")

# --------------------
# Example aggregations
# --------------------

# 1. Aggregate by rundir + dtype
agg_rundir_dtype = (
    df.groupby(["rundir", "dtype"])
    .agg(
        total=("label_mismatch", "count"),
        mismatches=("label_mismatch", "sum"),
        mismatch_rate=("label_mismatch", "mean"),
    )
    .reset_index()
)

# 2. Aggregate by rundir + dtype + platform
agg_rundir_dtype_platform = (
    df_long.groupby(["rundir", "dtype", "platform"])
    .agg(
        total=("label_mismatch", "count"),
        mismatches=("label_mismatch", "sum"),
        mismatch_rate=("label_mismatch", "mean"),
    )
    .reset_index()
)

# --------------------
# Output examples
# --------------------
print("\nAggregation by rundir + dtype:")
print(agg_rundir_dtype)

print("\nAggregation by rundir + dtype + platform:")
print(agg_rundir_dtype_platform)
