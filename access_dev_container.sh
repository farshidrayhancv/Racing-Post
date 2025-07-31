#!/bin/bash

# Smart script to connect to VS Code dev container
# Automatically detects dev container for the current project

# Configuration - Updated for your exact setup
PROJECT_NAME="post_race_analysis"  # This matches your project folder name
WORKDIR="/workspaces/Post_race_analysis"  # This is where VS Code mounts your project
LOCAL_PROJECT_PATH="$HOME/proj/Post_race_analysis"  # Your actual project location

# Function to find dev container
find_devcontainer() {
    # Method 1: Look for containers with project name in image
    CONTAINER_ID=$(docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Names}}" | \
        grep -i "$(echo $PROJECT_NAME | tr '[:upper:]' '[:lower:]' | tr '_' '-')" | head -1 | awk '{print $1}')
    
    if [ -n "$CONTAINER_ID" ]; then
        echo "$CONTAINER_ID"
        return 0
    fi
    
    # Method 2: Look for VS Code dev container patterns with project name
    CONTAINER_ID=$(docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Names}}" | \
        grep -E "vsc-.*$(echo $PROJECT_NAME | tr '[:upper:]' '[:lower:]' | tr '_' '-').*-features" | head -1 | awk '{print $1}')
    
    if [ -n "$CONTAINER_ID" ]; then
        echo "$CONTAINER_ID"
        return 0
    fi
    
    # Method 3: Look for any VS Code dev container if above methods fail
    CONTAINER_ID=$(docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Names}}" | \
        grep -E "vsc-.*-features" | head -1 | awk '{print $1}')
    
    if [ -n "$CONTAINER_ID" ]; then
        echo "$CONTAINER_ID"
        return 0
    fi
    
    return 1
}

# Function to get container name from ID
get_container_name() {
    docker ps --format "table {{.ID}}\t{{.Names}}" | grep "$1" | awk '{print $2}'
}

# Main execution
echo "🔍 Searching for dev container..."

if CONTAINER_ID=$(find_devcontainer); then
    CONTAINER_NAME=$(get_container_name "$CONTAINER_ID")
    echo "✅ Found dev container: $CONTAINER_NAME ($CONTAINER_ID)"
    echo "📁 Connecting to: $WORKDIR"
    echo ""
    
    # Connect to container as the vscode user (not root)
    docker exec -it -u vscode "$CONTAINER_ID" /bin/bash -c "cd $WORKDIR && exec /bin/bash"
else
    echo "❌ No dev container found!"
    echo ""
    echo "Make sure your dev container is running:"
    echo "1. Open VS Code"
    echo "2. Open this project: $CURRENT_DIR"
    echo "3. When prompted, click 'Reopen in Container'"
    echo "   Or use Ctrl+Shift+P > 'Dev Containers: Reopen in Container'"
    echo ""
    echo "Current running containers:"
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
    exit 1
fi