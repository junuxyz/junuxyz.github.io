import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/llm-engine/",
  plugins: [react()],
  test: {
    include: ["src/**/*.test.ts"],
    environment: "jsdom",
    coverage: {
      reporter: ["text", "html"]
    }
  }
});
