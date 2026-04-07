#!/usr/bin/env python3
"""Serve the landing page on port 8080."""
import http.server
import socketserver
from pathlib import Path

PORT = 8080
LANDING = Path(__file__).parent.parent / "src" / "dashboard" / "landing.html"


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(LANDING.read_bytes())


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Landing page: http://localhost:{PORT}")
        print(f"🌐 External: http://220.122.161.122:{PORT}")
        httpd.serve_forever()
