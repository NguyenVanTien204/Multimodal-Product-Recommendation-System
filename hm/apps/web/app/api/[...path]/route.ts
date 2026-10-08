import { NextRequest, NextResponse } from "next/server";

const BACKEND_BASE = process.env.DATN_API_BASE_URL || "http://localhost:8000";

async function proxyHandler(req: NextRequest, { params }: { params: { path: string[] } }) {
  const path = params.path ? params.path.join("/") : "";
  const targetUrl = new URL(`${BACKEND_BASE}/${path}${req.nextUrl.search}`);

  const headers = new Headers(req.headers);
  headers.delete("host");

  try {
    const body = ["GET", "HEAD"].includes(req.method) ? undefined : await req.arrayBuffer();

    const response = await fetch(targetUrl.toString(), {
      method: req.method,
      headers,
      body,
      cache: "no-store",
    });

    const responseHeaders = new Headers(response.headers);
    responseHeaders.delete("content-encoding");

    return new NextResponse(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error(`Failed to proxy to backend ${targetUrl}:`, error);
    return NextResponse.json(
      { error: "Backend proxy error", detail: String(error) },
      { status: 502 }
    );
  }
}

export const GET = proxyHandler;
export const POST = proxyHandler;
export const PUT = proxyHandler;
export const PATCH = proxyHandler;
export const DELETE = proxyHandler;
