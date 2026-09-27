import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep Turbopack inside this project. Without this it follows a parent
  // lockfile and can attempt to scan inaccessible user-profile directories.
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
