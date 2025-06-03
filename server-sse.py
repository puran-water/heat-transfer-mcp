#!/usr/bin/env python3
"""
HTTP/SSE transport entrypoint for Heat Transfer MCP Server
This script imports the existing FastMCP instance and runs it with HTTP transport for Modal deployment
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the existing FastMCP instance from server.py
try:
    from server import mcp
    print("✅ Successfully imported FastMCP instance from server.py", file=sys.stderr)
except ImportError as e:
    print(f"❌ Failed to import FastMCP instance: {e}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting Heat Transfer MCP Server with HTTP transport...", file=sys.stderr)
    
    # Use streamable-http transport (preferred) or sse (legacy)
    # Port 8000 is standard, host 0.0.0.0 allows external connections
    try:
        mcp.run(
            transport="streamable-http",
            port=8000,
            host="0.0.0.0"
        )
    except Exception as e:
        print(f"❌ Failed to start HTTP server: {e}", file=sys.stderr)
        print("🔄 Falling back to SSE transport...", file=sys.stderr)
        try:
            mcp.run(
                transport="sse",
                port=8000,
                host="0.0.0.0"
            )
        except Exception as e2:
            print(f"❌ Failed to start SSE server: {e2}", file=sys.stderr)
            sys.exit(1)