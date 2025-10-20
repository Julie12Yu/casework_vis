import json
from collections import defaultdict

FULL_PATH = "./public/3d_embedding.json"
RECLUSTER_PATH = "./public/3d_embedding_cluster3.json"
OUT_PATH = "merged_embedding.json"
CLUSTER_VALUE = 3

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def is_int_like(x):
    try:
        int(x)
        return True
    except Exception:
        return False

def to_int_if_possible(x):
    return int(x) if is_int_like(x) else x

def normalize_labels(labels):
    return [to_int_if_possible(v) for v in labels]

def find_cluster_indices(labels, cluster_value=3):
    idxs = []
    for i, lab in enumerate(labels):
        if (lab == cluster_value) or (is_int_like(lab) and int(lab) == cluster_value):
            idxs.append(i)
    return idxs

def title_index(titles):
    d = defaultdict(list)
    for i, t in enumerate(titles):
        d[t].append(i)
    return d

def remap_labels_away_from(existing_labels_outside_cluster, recluster_labels):
    """
    Make sure recluster labels don't collide with existing ones.
    We take max int label outside the big cluster + 1 as the base.
    """
    ints = [int(l) for l in existing_labels_outside_cluster if is_int_like(l)]
    base = (max(ints) + 1) if ints else 0

    remap = {}
    next_label = base
    mapped = []
    for l in recluster_labels:
        key = int(l) if is_int_like(l) else str(l)
        if key == -1:
            mapped.append(-1)
            continue
        if key not in remap:
            remap[key] = next_label
            next_label += 1
        mapped.append(remap[key])
    return mapped, remap

def main():
    full = load_json(FULL_PATH)
    recl = load_json(RECLUSTER_PATH)

    # Basic validations
    for name, obj in [("full", full), ("recluster", recl)]:
        for key in ["points", "labels", "titles", "summaries"]:
            if key not in obj:
                raise ValueError(f"'{key}' missing in {name} file.")
        n = len(obj["points"])
        if not (len(obj["labels"]) == len(obj["titles"]) == len(obj["summaries"]) == n):
            raise ValueError(f"In {name}, lengths of points/labels/titles/summaries do not match.")

    full_labels = normalize_labels(full["labels"])
    recl_labels = normalize_labels(recl["labels"])

    # indices of the big cluster in the full file
    c_idxs = find_cluster_indices(full_labels, CLUSTER_VALUE)

    # prepare label remapping for the reclustered labels to avoid collisions
    non_cluster_labels = [full_labels[i] for i in range(len(full_labels)) if i not in c_idxs]
    mapped_re_labels, remap = remap_labels_away_from(non_cluster_labels, recl_labels)

    # build title maps
    full_title_to_idxs = title_index(full["titles"])

    # start merged as a shallow copy of full
    merged = {
        "points": list(full["points"]),
        "labels": list(full_labels),
        "titles": list(full["titles"]),
        "summaries": list(full["summaries"]),
    }

    # choose which cluster-3 indices we can replace (by title)
    cluster_idx_set = set(c_idxs)
    used_full_positions = set()
    re_to_full = {}

    for re_i, t in enumerate(recl["titles"]):
        candidates = [idx for idx in full_title_to_idxs.get(t, []) if idx in cluster_idx_set and idx not in used_full_positions]
        if candidates:
            target = candidates[0]
            re_to_full[re_i] = target
            used_full_positions.add(target)

    # replace matched entries
    for re_i, target in re_to_full.items():
        merged["points"][target] = recl["points"][re_i]
        merged["labels"][target] = mapped_re_labels[re_i]
        merged["titles"][target] = recl["titles"][re_i]
        merged["summaries"][target] = recl["summaries"][re_i]

    # delete any leftover cluster-3 entries that weren't matched
    unused_cluster_idxs = sorted(cluster_idx_set - set(re_to_full.values()))
    for idx in reversed(unused_cluster_idxs):
        for key in ["points", "labels", "titles", "summaries"]:
            del merged[key][idx]

    # append recluster entries that didn't find a match
    unmatched_re = [i for i in range(len(recl["points"])) if i not in re_to_full]
    for re_i in unmatched_re:
        merged["points"].append(recl["points"][re_i])
        merged["labels"].append(mapped_re_labels[re_i])
        merged["titles"].append(recl["titles"][re_i])
        merged["summaries"].append(recl["summaries"][re_i])

    # ensure labels are plain ints where possible
    merged["labels"] = [to_int_if_possible(l) for l in merged["labels"]]

    # pretty-print JSON (spread out)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Saved pretty JSON to: {OUT_PATH}")

if __name__ == "__main__":
    main()