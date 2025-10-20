import json

input_file = "./public/3d_embedding.json"
output_file = "cluster_three_embed.json"

with open(input_file, "r") as f:
    data = json.load(f)

filtered = {
    title: [points, summary]
    for title, summary, label, points in zip(data["titles"], data["summaries"], data["labels"], data["points"])
    if label in {3}
}

with open(output_file, "w") as f:
    json.dump(filtered, f, indent=2)

print(f"Filtered cases saved to {output_file}")