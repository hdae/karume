// A WGSL file imported as text (`?raw` — Vite and Vitest both load it), for the baseline WGSL the
// state-attention check compares against. The playground's own types (vite-plus/client) declare the
// same suffix; this declaration covers the repository's type check.
declare module "*.wgsl?raw" {
  const source: string;
  export default source;
}
