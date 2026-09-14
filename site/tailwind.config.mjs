/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    container: {
      center: true,
      padding: {
        DEFAULT: "1.25rem",
        sm: "1.5rem",
        lg: "2.5rem",
        xl: "3rem",
        "2xl": "4rem",
      },
      screens: {
        "2xl": "1280px",
      },
    },
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        display: [
          "Inter Tight",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
      colors: {
        ink: {
          50: "#f7f7f8",
          100: "#eeeef0",
          200: "#d6d6da",
          300: "#b1b1b8",
          400: "#85858e",
          500: "#62626b",
          600: "#48484f",
          700: "#34343a",
          800: "#1d1d22",
          900: "#111114",
          950: "#0a0a0d",
        },
        accent: {
          50: "#f3f0ff",
          100: "#e9e3ff",
          200: "#d2c7ff",
          300: "#b39eff",
          400: "#8e6cff",
          500: "#7c5cff",
          600: "#6a40ff",
          700: "#5a2df0",
          800: "#4b25c4",
          900: "#3f219f",
          950: "#251359",
        },
        cyan: {
          400: "#22d3ee",
          500: "#06b6d4",
        },
      },
      letterSpacing: {
        tightest: "-0.04em",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.04), 0 30px 80px -20px rgba(124,92,255,0.45)",
        ring: "inset 0 0 0 1px rgba(255,255,255,0.06)",
        "ring-strong": "inset 0 0 0 1px rgba(255,255,255,0.1)",
        card: "0 1px 0 rgba(255,255,255,0.05) inset, 0 30px 60px -30px rgba(0,0,0,0.6)",
      },
      backgroundImage: {
        "grid-soft":
          "linear-gradient(to right, rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.04) 1px, transparent 1px)",
        "radial-fade":
          "radial-gradient(ellipse at top, rgba(124,92,255,0.18), transparent 60%)",
        "noise":
          "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.18 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")",
      },
      backgroundSize: {
        "grid-32": "32px 32px",
      },
      animation: {
        "fade-up": "fadeUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both",
        "fade-in": "fadeIn 0.6s ease-out both",
        "shimmer": "shimmer 2.6s linear infinite",
        "drift-slow": "drift 18s ease-in-out infinite",
        "drift-slower": "drift 28s ease-in-out infinite reverse",
        "orbit": "orbit 30s linear infinite",
        "pulse-soft": "pulseSoft 4s ease-in-out infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(14px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        drift: {
          "0%, 100%": { transform: "translate3d(0, 0, 0) scale(1)" },
          "50%": { transform: "translate3d(20px, -20px, 0) scale(1.05)" },
        },
        orbit: {
          "0%": { transform: "rotate(0deg) translateX(40px) rotate(0deg)" },
          "100%": { transform: "rotate(360deg) translateX(40px) rotate(-360deg)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "0.55" },
          "50%": { opacity: "0.85" },
        },
      },
    },
  },
  plugins: [],
};