import json

input_file = "3d_embedding.json"
output_file = "problem_clusters_unpacked.json"

with open(input_file, "r") as f:
    data = json.load(f)

filtered = {
    title: summary
    for title, summary, label in zip(data["titles"], data["summaries"], data["labels"])
    if label in {3, -1, 7}
}

with open(output_file, "w") as f:
    json.dump(filtered, f, indent=2)

print(f"Filtered cases saved to {output_file}")