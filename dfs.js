// Depth-First Search (DFS) implementations in JavaScript

// Graph represented as an adjacency list
// Example:
//   const graph = {
//     A: ['B', 'C'],
//     B: ['A', 'D', 'E'],
//     C: ['A', 'F'],
//     D: ['B'],
//     E: ['B', 'F'],
//     F: ['C', 'E'],
//   };

/**
 * DFS - Recursive
 * Visits all nodes reachable from `start`.
 * @param {Object} graph - adjacency list
 * @param {string} start - starting node
 * @param {Set} visited - (optional) set of already-visited nodes
 * @param {Array} order - (optional) traversal order accumulator
 * @returns {Array} order of visited nodes
 */
function dfsRecursive(graph, start, visited = new Set(), order = []) {
  visited.add(start);
  order.push(start);

  for (const neighbor of graph[start] || []) {
    if (!visited.has(neighbor)) {
      dfsRecursive(graph, neighbor, visited, order);
    }
  }

  return order;
}

/**
 * DFS - Iterative (explicit stack)
 * @param {Object} graph - adjacency list
 * @param {string} start - starting node
 * @returns {Array} order of visited nodes
 */
function dfsIterative(graph, start) {
  const visited = new Set();
  const stack = [start];
  const order = [];

  while (stack.length > 0) {
    const node = stack.pop();
    if (visited.has(node)) continue;

    visited.add(node);
    order.push(node);

    // Push neighbors in reverse so they're visited in original order
    const neighbors = graph[node] || [];
    for (let i = neighbors.length - 1; i >= 0; i--) {
      if (!visited.has(neighbors[i])) {
        stack.push(neighbors[i]);
      }
    }
  }

  return order;
}

/**
 * DFS - Find a path from start to target
 * @param {Object} graph - adjacency list
 * @param {string} start - starting node
 * @param {string} target - target node
 * @returns {Array|null} path from start to target, or null if unreachable
 */
function dfsFindPath(graph, start, target) {
  const visited = new Set();
  const path = [];

  function search(node) {
    visited.add(node);
    path.push(node);

    if (node === target) return true;

    for (const neighbor of graph[node] || []) {
      if (!visited.has(neighbor) && search(neighbor)) {
        return true;
      }
    }

    path.pop(); // backtrack
    return false;
  }

  return search(start) ? path : null;
}

// ---- Example usage ----
const graph = {
  A: ['B', 'C'],
  B: ['A', 'D', 'E'],
  C: ['A', 'F'],
  D: ['B'],
  E: ['B', 'F'],
  F: ['C', 'E'],
};

console.log('Recursive DFS from A:', dfsRecursive(graph, 'A').join(' -> '));
console.log('Iterative DFS from A:', dfsIterative(graph, 'A').join(' -> '));
console.log('Path A -> F:', dfsFindPath(graph, 'A', 'F')?.join(' -> ') ?? 'not found');

module.exports = { dfsRecursive, dfsIterative, dfsFindPath };
