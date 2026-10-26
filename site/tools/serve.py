"""Serve the project page locally with explicit ES-module MIME types."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8787)
    a=p.parse_args()
    SimpleHTTPRequestHandler.extensions_map.update({'.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml'})
    handler=partial(SimpleHTTPRequestHandler,directory=str(Path(__file__).resolve().parents[1]))
    print(f'H-GBP preview: http://127.0.0.1:{a.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),handler).serve_forever()
