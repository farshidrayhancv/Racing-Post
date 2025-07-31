from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import time
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import uuid

# Import your main racing analysis code
from main import EnhancedRacingAgent, setup_model_client

app = FastAPI(title="Horse Racing Commentary API")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store active connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_data: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.session_data[session_id] = {"websocket": websocket, "status": "connected"}

    def disconnect(self, websocket: WebSocket, session_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if session_id in self.session_data:
            del self.session_data[session_id]

    async def send_update(self, session_id: str, message: dict):
        if session_id in self.session_data:
            websocket = self.session_data[session_id]["websocket"]
            try:
                await websocket.send_text(json.dumps(message))
            except:
                self.disconnect(websocket, session_id)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/config")
async def get_config():
    """Serve config values for form pre-population"""
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        # Return all values for convenience
        return {
            "model_provider": config.get("model_provider", "anthropic"),
            "anthropic_api_key": config.get("anthropic_api_key", ""),
            "deepseek_api_key": config.get("deepseek_api_key", ""),
            "tpd_api_key": config.get("tpd_api_key", ""),
            "tpd_live_recording_api_key": config.get("tpd_live_recording_api_key", ""),
            "race_sharecode": config.get("race_sharecode", "01202506191430")
        }
    except FileNotFoundError:
        return {"race_sharecode": "01202506191430"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)

@app.post("/analyze")
async def analyze_race(config: dict):
    """Process race analysis with real-time updates"""
    session_id = str(uuid.uuid4())
    
    # Start async processing
    asyncio.create_task(process_race_analysis(config, session_id))
    
    return {"session_id": session_id, "status": "processing"}

async def process_race_analysis(config: Dict, session_id: str):
    """Main processing function with progress updates"""
    try:
        # Step 1: Model setup
        await manager.send_update(session_id, {
            "type": "progress",
            "step": "setup",
            "message": "🤖 Setting up AI model...",
            "progress": 10
        })
        
        # Validate configuration
        required_keys = ['tpd_api_key', 'tpd_live_recording_api_key', 'race_sharecode']
        missing_keys = [key for key in required_keys if not config.get(key)]
        
        if missing_keys:
            await manager.send_update(session_id, {
                "type": "error",
                "message": f"Missing required keys: {', '.join(missing_keys)}"
            })
            return
        
        # Initialize agent
        try:
            # Initialize agent
            print(f"🔧 Initializing agent with provider: {config.get('model_provider')}")
            agent = EnhancedRacingAgent(
                config, 
                config['tpd_api_key'], 
                config['tpd_live_recording_api_key']
            )
            print(f"✅ Agent initialized successfully")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            await manager.send_update(session_id, {
                "type": "error",
                "message": f"❌ Model initialization failed: {str(e)}"
            })
            return
        
        await manager.send_update(session_id, {
            "type": "progress",
            "step": "model_ready",
            "message": f"✅ {agent.provider.upper()} model ready: {agent.model_name}",
            "progress": 20
        })
        
        # Step 2: Fetch TPD data
        await manager.send_update(session_id, {
            "type": "progress",
            "step": "fetch_tpd",
            "message": "📊 Fetching performance data...",
            "progress": 30
        })
        
        tpd_data = agent.tpd_client.get_performance_metrics(config['race_sharecode'])
        
        if not tpd_data.get('success'):
            await manager.send_update(session_id, {
                "type": "error",
                "message": "❌ Failed to fetch TPD performance data"
            })
            return
            
        # Step 3: Fetch GPS data with progress feedback
        await manager.send_update(session_id, {
            "type": "progress",
            "step": "fetch_gps",
            "message": "📡 Fetching GPS position data...",
            "progress": 40
        })
        
        s3_url = agent.tpd_client.get_live_recording_url(config['race_sharecode'])
        if not s3_url:
            await manager.send_update(session_id, {
                "type": "error",
                "message": "❌ Could not fetch GPS data URL"
            })
            return
            
        # Add progress update for large GPS download
        await manager.send_update(session_id, {
            "type": "progress",
            "step": "download_gps",
            "message": "📥 Downloading GPS data (this may take a moment for large races)...",
            "progress": 42
        })
        
        gps_records = agent.tpd_client.fetch_gps_data(s3_url)
        
        # Process GPS records with progress feedback
        await manager.send_update(session_id, {
            "type": "progress",
            "step": "process_gps",
            "message": f"🔄 Processing {len(gps_records):,} GPS records...",
            "progress": 45
        })
        
        # Allow event loop to breathe during heavy processing
        await asyncio.sleep(0.1)
        
        gps_position_data = agent.tpd_client.extract_gps_position_data(gps_records)
        
        await manager.send_update(session_id, {
            "type": "progress",
            "step": "gps_loaded",
            "message": f"✅ GPS data loaded: {len(gps_records):,} records, {len(gps_position_data)} markers",
            "progress": 50
        })
        
        # Step 4: CRITICAL FIX - Use NEW enhanced method signature
        performance_rankings, horse_assignments = agent.find_top_performance_horses(tpd_data)
        
        # Count assigned horses for progress message
        assigned_count = len(horse_assignments)
        
        await manager.send_update(session_id, {
            "type": "progress",
            "step": "analysis_start",
            "message": f"🏆 Starting analysis with {assigned_count} horses assigned performance metrics...",
            "progress": 60
        })
        
        # Step 5: Process horses
        commentaries = {}
        tpd_runners = tpd_data.get('runners', {})
        total_horses = len([r for r in tpd_runners.values() if r.get('position') and r.get('position') not in ['NR', 'W']])
        processed = 0
        
        for horse_id, runner_info in tpd_runners.items():
            horse_name = runner_info.get('horse', f"Horse {horse_id}")
            position = runner_info.get('position')
            
            if not position or position in ['NR', 'W']:
                continue
                
            # Show if horse has assigned metric in progress
            metric_info = ""
            if horse_id in horse_assignments:
                assigned = horse_assignments[horse_id]
                metric_info = f" ({assigned['rank_text']} {assigned['display_name']})"
                
            await manager.send_update(session_id, {
                "type": "progress",
                "step": "processing_horse",
                "message": f"🔄 Processing {horse_name}{metric_info}...",
                "horse_name": horse_name,
                "progress": 60 + (processed / total_horses) * 30
            })
            
            try:
                final_position = int(position)
            except (ValueError, TypeError):
                final_position = 999
                
            full_horse_id = f"{config['race_sharecode']}{horse_id}"
            race_data = {
                'horse_id': full_horse_id,
                'horse_name': horse_name,
                'final_position': final_position
            }
            
            try:
                commentary = await agent.process_horse(race_data, tpd_data, gps_position_data, horse_assignments)
                
                # Include performance metric info in results
                commentary_data = {
                    'commentary': commentary,
                    'position': final_position,
                    'horse_id': int(horse_id),
                    'performance': runner_info
                }
                
                # Add assigned metric info if exists
                if horse_id in horse_assignments:
                    commentary_data['assigned_metric'] = horse_assignments[horse_id]
                
                commentaries[horse_name] = commentary_data
                
                await manager.send_update(session_id, {
                    "type": "horse_complete",
                    "horse_name": horse_name,
                    "commentary": commentary,
                    "position": final_position,
                    "assigned_metric": horse_assignments.get(horse_id)
                })
                
            except Exception as e:
                await manager.send_update(session_id, {
                    "type": "horse_error",
                    "horse_name": horse_name,
                    "error": str(e)
                })
                
            processed += 1
            await asyncio.sleep(1)  # Brief pause between horses
        
        # Step 6: Final results with enhanced metrics info
        await manager.send_update(session_id, {
            "type": "complete",
            "message": "✅ Analysis complete!",
            "progress": 100,
            "results": commentaries,
            "race_info": {
                "sharecode": config['race_sharecode'],
                "total_runners": total_horses,
                "model": agent.model_name,
                "provider": agent.provider.upper(),
                "metrics_assigned": assigned_count
            }
        })
        
    except Exception as e:
        await manager.send_update(session_id, {
            "type": "error",
            "message": f"❌ Processing failed: {str(e)}"
        })

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}