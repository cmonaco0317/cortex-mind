import { defineConfig } from "vite";

// Cortex is a single-page static app (index.html). `base` is passed at build time
// for project-page hosting, e.g. GitHub Pages: `npm run build -- --base=/cortex-mind/`.
export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
  },
});
