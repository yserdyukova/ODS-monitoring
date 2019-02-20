#!/opt/anaconda3/bin/python3

import http.server
import socketserver

PORT = 8003

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()