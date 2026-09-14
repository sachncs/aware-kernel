import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

const SITE_URL = "https://sachncs.github.io/kernos";

export default defineConfig({
  site: SITE_URL,
  base: "/kernos",
  output: "static",
  integrations: [
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  build: {
    inlineStylesheets: "auto",
  },
});