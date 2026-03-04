#!/bin/bash
ray stop --force 2>/dev/null
pkill -9 -f "ray::" 2>/dev/null
pkill -9 -f "main_task" 2>/dev/null
pkill -9 -f "gcs_server" 2>/dev/null
rm -rf /tmp/ray 2>/dev/null
echo "Ray processes cleaned up."
