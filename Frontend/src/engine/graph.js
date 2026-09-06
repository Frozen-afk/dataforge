// Graph primitives and the independent BFS oracle.
//
// This is a direct port of Backend/core/bfs.py and the adjacency helpers in
// Backend/core/graph.py. It runs in the browser so the lesson works from a
// static URL with no server, and so the depth slider redraws in the same frame
// the learner moves it.
//
// The oracle stays independent of the recurrence in exactly the sense the
// backend keeps it independent: nothing in this file imports from exact.js,
// and nothing in exact.js imports the distance from here except to *compare*
// against it. Two procedures agreeing is evidence; one procedure agreeing with
// itself is not.

export function outgoing(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) adjacency[u].push(v);
  return adjacency;
}

export function incoming(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) adjacency[v].push(u);
  return adjacency;
}

/**
 * Shortest-path distance from source to target, or null if unreachable.
 * Breadth-first search over the outgoing adjacency list.
 */
export function bfsDistance(n, edges, source, target) {
  if (source === target) return 0;

  const adjacency = outgoing(n, edges);
  const distance = new Array(n).fill(-1);
  distance[source] = 0;

  const queue = [source];
  let head = 0;

  while (head < queue.length) {
    const u = queue[head++];
    for (const v of adjacency[u]) {
      if (distance[v] === -1) {
        distance[v] = distance[u] + 1;
        if (v === target) return distance[v];
        queue.push(v);
      }
    }
  }

  return null;
}

/** BFS layer index of every node, or null for nodes the source cannot reach. */
export function bfsLayers(n, edges, source) {
  const adjacency = outgoing(n, edges);
  const layer = new Array(n).fill(null);
  layer[source] = 0;

  const queue = [source];
  let head = 0;

  while (head < queue.length) {
    const u = queue[head++];
    for (const v of adjacency[u]) {
      if (layer[v] === null) {
        layer[v] = layer[u] + 1;
        queue.push(v);
      }
    }
  }

  return layer;
}

/**
 * One shortest path from source to target as a node list, or null.
 * Used only to highlight the path in the drawing; no measurement reads it.
 */
export function shortestPath(n, edges, source, target) {
  const adjacency = outgoing(n, edges);
  const parent = new Array(n).fill(-1);
  const seen = new Array(n).fill(false);
  seen[source] = true;

  const queue = [source];
  let head = 0;

  while (head < queue.length) {
    const u = queue[head++];
    if (u === target) break;
    for (const v of adjacency[u]) {
      if (!seen[v]) {
        seen[v] = true;
        parent[v] = u;
        queue.push(v);
      }
    }
  }

  if (!seen[target]) return null;

  const path = [target];
  while (path[0] !== source) path.unshift(parent[path[0]]);
  return path;
}

/** Node-count and edge-count summary used by the honesty panels. */
export function graphStats(graph, source, target) {
  const inDegree = graph.edges.filter(([, v]) => v === target).length;
  const outDegree = graph.edges.filter(([u]) => u === source).length;
  return {
    numNodes: graph.n,
    numEdges: graph.edges.length,
    targetInDegree: inDegree,
    sourceOutDegree: outDegree,
  };
}
