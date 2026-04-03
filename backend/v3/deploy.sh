#!/bin/bash
# Deploy Discovery Social API (v3) to Modal
# Run from this directory: ./deploy.sh

set -e

echo "🎵 Deploying Discovery Social API v3..."
echo ""
echo "This will deploy:"
echo "  - App: discovery-social"
echo "  - Volumes: discovery-social-data (new), discovery-fragments (existing)"
echo "  - Endpoints: /tracks, /agents, /health"
echo ""

modal deploy app.py

echo ""
echo "✅ Deployed! Your API is live at:"
echo "   https://heyitskim-ai--discovery-social-web.modal.run"
echo ""
echo "Test it:"
echo "   curl https://heyitskim-ai--discovery-social-web.modal.run/health"
