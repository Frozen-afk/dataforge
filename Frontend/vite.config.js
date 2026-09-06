import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The artifact has to open from a plain static URL with no server behind it,
// which means it may be served from a subpath (GitHub Pages project sites, for
// example). BASE_PATH lets a deployment say so at build time without touching
// any source file; everything that loads a data file goes through
// import.meta.env.BASE_URL, so setting it here is enough.
//
//   npm run build                      -> served from /
//   BASE_PATH=/dataforge/ npm run build -> served from /dataforge/
export default defineConfig({
  base: process.env.BASE_PATH || "/",
  plugins: [react()],
  build: {
    outDir: "dist",
    // The bundle is one page with no routing, so a single chunk loads faster
    // than several round trips. The data files are fetched separately and are
    // the only thing that grows.
    chunkSizeWarningLimit: 700,
  },
});
