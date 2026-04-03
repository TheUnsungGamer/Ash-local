import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      "/lmstudio/models": {
        target: "http://127.0.0.1:1234",
        changeOrigin: true,
        rewrite: () => "/v1/models",
      },
      "/lmstudio/chat/completions": {
        target: "http://127.0.0.1:1234",
        changeOrigin: true,
        rewrite: () => "/v1/chat/completions",
      },
    },
  },
});