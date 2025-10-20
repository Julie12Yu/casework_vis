**How run?**

- npm install
- npx vite

**Purpose**

- Get a working visualization for casework project

---

**TODOS**

- Set a min-max on where the camera can go, ensure cannot go to blank screen
- Set more and easier default camera positions
- Add instructions for using the graph
- Get cluster categories for the summaries
- Automatically zoom towards the point/cluster you click on

**Potential TODO**

- Add view for focusing on inside of the clusters
- Add view for focusing on outside of the clusters

**Embedding Structure**
{
"points": [[x1, y1, z1], [x2, y2, z2], ...],
"labels":[point1num, point2num, ...],
"titles":["title", "title2", ...],
"summaries":["summary", "summary2", ...]
}
