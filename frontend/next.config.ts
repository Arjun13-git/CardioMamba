import type { NextConfig } from "next";

// The FastAPI inference service (backend/). Requests to /api/v1/* are proxied to it so the
// browser talks to a single origin (no CORS setup needed for the demo).
const API_URL = process.env.CARDIOMAMBA_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/v1/:path*", destination: `${API_URL}/api/v1/:path*` }];
  },
};

export default nextConfig;
