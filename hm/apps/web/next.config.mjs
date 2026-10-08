/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    // Shop photos are already 512 px JPEGs served by our backend (/api/static/hm/...): no optimizer needed, and
    // sharp resizing would only burn CPU on the 2-vCPU server.
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
      {
        protocol: 'http',
        hostname: '**',
      },
    ],
  },
};

export default nextConfig;
