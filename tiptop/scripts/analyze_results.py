"""Aggregate metrics from TiPToP server output directories."""

import argparse
import json
from collections import Counter
from pathlib import Path


def categorize_failure(reason: str) -> str:
    """Bucket a raw failure_reason string into a high-level category."""
    if not reason:
        return "unknown"
    if reason.startswith("Motion planning failed"):
        return "motion_planning_failed"
    if reason.startswith("No satisfying particles found after optimizing") and "time budget" in reason:
        return "timeout_no_satisfying_particles"
    if reason.startswith("No satisfying particles found after optimizing all"):
        return "no_satisfying_particles"
    if reason.startswith("All") and "failed particle initialization" in reason:
        return "particle_initialization_failed"
    if reason.startswith("No valid plan skeletons"):
        return "no_valid_plan_skeletons"
    if reason.startswith("No plane found with objects resting"):
        return "no_plane_found"
    if "[Errno" in reason:
        return "file_error"
    return reason  # fall through — keep raw string as its own bucket


def analyze_directory(root: Path) -> None:
    """Analyze all run subdirectories under *root* and print aggregate metrics."""
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    total = len(subdirs)

    success_count = 0
    failure_count = 0
    missing_metadata_count = 0

    failure_reasons: Counter[str] = Counter()
    failure_details: dict[str, list[str]] = {}  # category -> list of raw reasons
    failure_atoms: dict[str, list[list[dict]]] = {}  # category -> grounded atoms per run
    failure_runs: dict[str, list[Path]] = {}  # category -> list of run directories
    all_grounded_atoms: list[list[dict]] = []

    # Categories where we want to inspect grounded atoms on failure
    atoms_inspect_categories = {"no_valid_plan_skeletons", "particle_initialization_failed"}

    for d in subdirs:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            missing_metadata_count += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Grounded atoms
        atoms = meta.get("perception", {}).get("grounded_atoms")
        if atoms is not None:
            all_grounded_atoms.append(atoms)

        # Planning success
        planning = meta.get("planning", {})
        if planning.get("success"):
            success_count += 1
        else:
            failure_count += 1
            raw_reason = planning.get("failure_reason") or "unknown"
            category = categorize_failure(raw_reason)
            failure_reasons[category] += 1
            failure_details.setdefault(category, []).append(raw_reason)
            failure_runs.setdefault(category, []).append(d)
            if category in atoms_inspect_categories and atoms is not None:
                failure_atoms.setdefault(category, []).append(atoms)

    # --- Print report ---
    print(f"\n{'=' * 60}")
    print(f"Results for: {root}")
    print(f"{'=' * 60}")

    print(f"\n--- Overall ---")
    print(f"Total runs:            {total}")
    print(f"  Success:             {success_count} ({100 * success_count / total:.1f}%)")
    print(f"  Failure:             {failure_count} ({100 * failure_count / total:.1f}%)")
    print(f"  Missing metadata:    {missing_metadata_count} ({100 * missing_metadata_count / total:.1f}%)")

    if failure_reasons:
        print(f"\n--- Failures by category ---")
        for category, count in failure_reasons.most_common():
            print(f"  {category}: {count}")
            # Show unique raw reasons in this bucket (deduplicated)
            unique_raw = sorted(set(failure_details[category]))
            if len(unique_raw) == 1 and unique_raw[0] == category:
                continue  # no extra detail to show
            for raw in unique_raw[:5]:
                print(f"    e.g. {raw}")
            if len(unique_raw) > 5:
                print(f"    ... and {len(unique_raw) - 5} more variants")

    # Grounded atoms breakdown for key failure categories
    for category in atoms_inspect_categories:
        runs_atoms = failure_atoms.get(category, [])
        if not runs_atoms:
            continue
        print(f"\n--- Grounded atoms for '{category}' failures ({len(runs_atoms)} runs) ---")

        atom_counts_per_run = [len(a) for a in runs_atoms]
        print(f"  Atoms per run:  avg {sum(atom_counts_per_run) / len(atom_counts_per_run):.1f}"
              f"  min {min(atom_counts_per_run)}  max {max(atom_counts_per_run)}")

        # How many had zero atoms
        zero_atom_runs = sum(1 for c in atom_counts_per_run if c == 0)
        if zero_atom_runs:
            print(f"  Runs with 0 atoms: {zero_atom_runs}")

        cat_obj_counts: Counter[str] = Counter()
        cat_pred_counts: Counter[str] = Counter()
        cat_atom_strs: Counter[str] = Counter()
        for atoms in runs_atoms:
            for atom in atoms:
                cat_pred_counts[atom["predicate"]] += 1
                for arg in atom["args"]:
                    cat_obj_counts[arg] += 1
                atom_str = f"{atom['predicate']}({', '.join(atom['args'])})"
                cat_atom_strs[atom_str] += 1

        print(f"  Predicate distribution:")
        for pred, cnt in cat_pred_counts.most_common():
            print(f"    {pred}: {cnt}")
        print(f"  Top 10 atoms:")
        for atom_str, cnt in cat_atom_strs.most_common(10):
            print(f"    {atom_str}: {cnt}")
        print(f"  Top 10 objects:")
        for obj, cnt in cat_obj_counts.most_common(10):
            print(f"    {obj}: {cnt}")

    # List individual runs for no_valid_plan_skeletons
    skeleton_runs = failure_runs.get("no_valid_plan_skeletons", [])
    if skeleton_runs:
        print(f"\n--- 'no_valid_plan_skeletons' run directories ({len(skeleton_runs)}) ---")
        for d in skeleton_runs:
            meta = json.load(open(d / "metadata.json"))
            task = meta.get("task_instruction", "?")
            atoms = meta.get("perception", {}).get("grounded_atoms", [])
            atoms_str = ", ".join(f"{a['predicate']}({', '.join(a['args'])})" for a in atoms)
            print(f"  {d.name}  task={task!r}  atoms=[{atoms_str}]")

    # Grounded atoms stats
    if all_grounded_atoms:
        predicate_counts: Counter[str] = Counter()
        object_counts: Counter[str] = Counter()
        atoms_per_run = []
        for atoms in all_grounded_atoms:
            atoms_per_run.append(len(atoms))
            for atom in atoms:
                predicate_counts[atom["predicate"]] += 1
                for arg in atom["args"]:
                    object_counts[arg] += 1

        print(f"\n--- Grounded atoms ---")
        print(f"  Runs with atoms:     {len(all_grounded_atoms)}")
        print(f"  Avg atoms per run:   {sum(atoms_per_run) / len(atoms_per_run):.1f}")
        print(f"  Min / Max:           {min(atoms_per_run)} / {max(atoms_per_run)}")
        print(f"\n  Predicate distribution:")
        for pred, cnt in predicate_counts.most_common():
            print(f"    {pred}: {cnt}")
        print(f"\n  Top 15 objects:")
        for obj, cnt in object_counts.most_common(15):
            print(f"    {obj}: {cnt}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze TiPToP result directories.")
    parser.add_argument("dirs", nargs="+", type=Path, help="Root directories containing run subdirectories")
    args = parser.parse_args()

    for d in args.dirs:
        if not d.is_dir():
            print(f"WARNING: {d} is not a directory, skipping.")
            continue
        analyze_directory(d)


if __name__ == "__main__":
    main()
