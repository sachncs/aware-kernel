export const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
export const withBase = (path) => {
  if (!path) return BASE;
  if (path.startsWith("http") || path.startsWith("mailto:")) return path;
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `${BASE}${clean}`;
};