#!/bin/bash
# Kill all /bin/bash ./start_server.sh and python3 -m workers.comfyui.server processes, then restart start_server.sh in the background with no output.

# Kill /bin/bash ./start_server.sh processes
ps -ef | grep '/bin/bash ./start_server.sh' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Kill python3 -m workers.comfyui.server processes
ps -ef | grep 'python3 -m workers.comfyui.server' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Restart start_server.sh in the background, suppressing all output
nohup /bin/bash ./start_server.sh >/dev/null 2>&1 &
