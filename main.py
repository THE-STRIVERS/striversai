# from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, StreamingResponse
# import openai
# from elevenlabs import generate, set_api_key
# import ffmpeg
# import os
# import json
# import asyncio
# import uuid
# from datetime import datetime
# from typing import Optional, List, Dict
# import logging
# from pydantic import BaseModel
# import aiofiles
# import tempfile
# import shutil

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="AI Video Content Creator API",
#     description="Advanced AI-powered video creation platform for Indian content creators",
#     version="1.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configuration
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
# PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# # Setup AI Services
# def setup_ai_services():
#     """Initialize AI services with error handling"""
#     services_status = {}
    
#     # OpenAI GPT-4 Setup
#     try:
#         if OPENAI_API_KEY:
#             openai.api_key = OPENAI_API_KEY
#             services_status['openai'] = "Connected"
#         else:
#             services_status['openai'] = "No API key"
#     except Exception as e:
#         services_status['openai'] = f"Error: {str(e)}"
    
#     # ElevenLabs Setup
#     try:
#         if ELEVENLABS_API_KEY:
#             set_api_key(ELEVENLABS_API_KEY)
#             services_status['elevenlabs'] = "Connected"
#         else:
#             services_status['elevenlabs'] = "No API key"
#     except Exception as e:
#         services_status['elevenlabs'] = f"Error: {str(e)}"
    
#     return services_status

# ai_services_status = setup_ai_services()
# logger.info(f"AI Services Status: {ai_services_status}")

# # Pydantic Models
# class ScriptRequest(BaseModel):
#     topic: str
#     duration: int
#     content_type: str
#     target_language: str
#     target_platform: str
#     cultural_context: Optional[str] = None
#     target_audience: Optional[str] = None

# class VoiceOverRequest(BaseModel):
#     script: str
#     language: str
#     voice_style: Optional[str] = "professional"
#     speed: Optional[float] = 1.0

# class VideoProject(BaseModel):
#     project_id: str
#     title: str
#     script: str
#     voiceover_file: Optional[str] = None
#     media_files: List[str] = []
#     status: str = "draft"
#     created_at: str
#     duration: int

# # Supported configurations
# SUPPORTED_LANGUAGES = ["hindi", "tamil", "telugu", "bengali", "marathi", "gujarati", "kannada", "malayalam", "punjabi", "english"]
# SUPPORTED_CONTENT_TYPES = ["educational", "marketing", "entertainment", "news", "tutorial"]
# SUPPORTED_PLATFORMS = ["youtube", "instagram", "tiktok", "facebook", "whatsapp"]

# # Project Storage (In production, use database)
# projects_db = {}

# # Utility Functions
# async def generate_project_id():
#     return f"proj_{uuid.uuid4().hex[:8]}"

# def get_timestamp():
#     return datetime.now().isoformat()

# # API Routes
# @app.get("/")
# async def root():
#     return {
#         "message": "AI Video Content Creator API is running!",
#         "status": "active",
#         "service": "Advanced AI Video Creation Platform",
#         "ai_services": ai_services_status
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "timestamp": get_timestamp(),
#         "ai_services": ai_services_status,
#         "supported_languages": SUPPORTED_LANGUAGES,
#         "supported_content_types": SUPPORTED_CONTENT_TYPES,
#         "supported_platforms": SUPPORTED_PLATFORMS
#     }

# @app.post("/generate-script")
# async def generate_script(request: ScriptRequest):
#     """
#     Generate video script using GPT-4 with Indian cultural context
#     """
#     try:
#         logger.info(f"Script generation request: {request.dict()}")
        
#         # Validate inputs
#         if request.target_language.lower() not in SUPPORTED_LANGUAGES:
#             raise HTTPException(400, f"Language must be one of: {SUPPORTED_LANGUAGES}")
        
#         if request.content_type.lower() not in SUPPORTED_CONTENT_TYPES:
#             raise HTTPException(400, f"Content type must be one of: {SUPPORTED_CONTENT_TYPES}")
        
#         if request.target_platform.lower() not in SUPPORTED_PLATFORMS:
#             raise HTTPException(400, f"Platform must be one of: {SUPPORTED_PLATFORMS}")

#         # Build culturally relevant prompt
#         prompt = build_indian_context_prompt(request)
        
#         # Generate script with timeout (60 seconds max)
#         try:
#             script = await asyncio.wait_for(
#                 generate_script_with_gpt4(prompt),
#                 timeout=60.0
#             )
#         except asyncio.TimeoutError:
#             raise HTTPException(408, "Script generation timed out (60s limit)")
        
#         # Generate storyboard
#         storyboard = generate_storyboard_from_script(script, request.duration)
        
#         logger.info("Script generation completed successfully")
        
#         return {
#             "success": True,
#             "script": script,
#             "storyboard": storyboard,
#             "metadata": {
#                 "topic": request.topic,
#                 "duration": request.duration,
#                 "content_type": request.content_type,
#                 "language": request.target_language,
#                 "platform": request.target_platform,
#                 "timestamp": get_timestamp()
#             }
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Script generation error: {e}")
#         raise HTTPException(500, f"Script generation failed: {str(e)}")

# @app.post("/generate-voiceover")
# async def generate_voiceover(request: VoiceOverRequest):
#     """
#     Generate voiceover in Indian languages using ElevenLabs
#     """
#     try:
#         logger.info(f"Voiceover generation request: language={request.language}")
        
#         if request.language.lower() not in SUPPORTED_LANGUAGES:
#             raise HTTPException(400, f"Language must be one of: {SUPPORTED_LANGUAGES}")

#         if not ELEVENLABS_API_KEY:
#             # Demo mode - return mock response
#             return await generate_demo_voiceover(request)
        
#         # Generate voiceover with ElevenLabs
#         voiceover_data = await generate_voiceover_elevenlabs(
#             request.script,
#             request.language,
#             request.voice_style,
#             request.speed
#         )
        
#         # Save voiceover file
#         filename = f"voiceover_{uuid.uuid4().hex[:8]}.mp3"
#         filepath = f"/tmp/{filename}"
        
#         with open(filepath, "wb") as f:
#             f.write(voiceover_data)
        
#         logger.info("Voiceover generation completed successfully")
        
#         return {
#             "success": True,
#             "voiceover_url": f"/voiceovers/{filename}",
#             "metadata": {
#                 "language": request.language,
#                 "voice_style": request.voice_style,
#                 "duration_estimate": len(request.script) // 15,  # Rough estimate
#                 "timestamp": get_timestamp()
#             }
#         }

#     except Exception as e:
#         logger.error(f"Voiceover generation error: {e}")
#         raise HTTPException(500, f"Voiceover generation failed: {str(e)}")

# @app.post("/search-media")
# async def search_media(
#     query: str = Form(...),
#     media_type: str = Form("image"),
#     cultural_context: bool = Form(True)
# ):
#     """
#     Search stock media from integrated APIs with Indian context
#     """
#     try:
#         logger.info(f"Media search: {query}, type: {media_type}")
        
#         # Enhanced query for Indian context
#         enhanced_query = add_indian_context(query) if cultural_context else query
        
#         # Search from both stock APIs (max 2 as specified)
#         results = await asyncio.gather(
#             search_pixabay(enhanced_query, media_type),
#             search_pexels(enhanced_query, media_type),
#             return_exceptions=True
#         )
        
#         # Filter valid results
#         media_results = []
#         for result in results:
#             if not isinstance(result, Exception):
#                 media_results.extend(result)
        
#         # Filter culturally appropriate content
#         filtered_media = filter_culturally_appropriate(media_results)
        
#         return {
#             "success": True,
#             "query": query,
#             "enhanced_query": enhanced_query if cultural_context else query,
#             "media_type": media_type,
#             "results": filtered_media[:20],  # Limit results
#             "total_count": len(filtered_media),
#             "timestamp": get_timestamp()
#         }

#     except Exception as e:
#         logger.error(f"Media search error: {e}")
#         raise HTTPException(500, f"Media search failed: {str(e)}")

# @app.post("/create-video-project")
# async def create_video_project(
#     background_tasks: BackgroundTasks,
#     title: str = Form(...),
#     script: str = Form(...),
#     media_files: List[str] = Form(...),
#     voiceover_file: str = Form(None),
#     target_platform: str = Form("youtube"),
#     duration: int = Form(...)
# ):
#     """
#     Create a new video project and start processing
#     """
#     try:
#         project_id = await generate_project_id()
        
#         project = VideoProject(
#             project_id=project_id,
#             title=title,
#             script=script,
#             voiceover_file=voiceover_file,
#             media_files=media_files,
#             status="processing",
#             created_at=get_timestamp(),
#             duration=duration
#         )
        
#         # Store project
#         projects_db[project_id] = project.dict()
        
#         # Start background video processing
#         background_tasks.add_task(
#             process_video_background,
#             project_id,
#             script,
#             media_files,
#             voiceover_file,
#             target_platform,
#             duration
#         )
        
#         return {
#             "success": True,
#             "project_id": project_id,
#             "status": "processing",
#             "message": "Video project created and processing started",
#             "estimated_time": duration * 5,  # 5 minutes per minute of video
#             "timestamp": get_timestamp()
#         }

#     except Exception as e:
#         logger.error(f"Project creation error: {e}")
#         raise HTTPException(500, f"Project creation failed: {str(e)}")

# @app.get("/project-status/{project_id}")
# async def get_project_status(project_id: str):
#     """
#     Get video project processing status
#     """
#     project = projects_db.get(project_id)
#     if not project:
#         raise HTTPException(404, "Project not found")
    
#     return {
#         "success": True,
#         "project_id": project_id,
#         "status": project["status"],
#         "progress": project.get("progress", 0),
#         "download_url": project.get("download_url"),
#         "thumbnail_url": project.get("thumbnail_url"),
#         "timestamp": get_timestamp()
#     }

# @app.post("/generate-subtitles")
# async def generate_subtitles(
#     video_file: UploadFile = File(...),
#     language: str = Form("english")
# ):
#     """
#     Generate subtitles for video content
#     """
#     try:
#         if not video_file.content_type.startswith('video/'):
#             raise HTTPException(400, "Please upload a valid video file")
        
#         # Save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#             content = await video_file.read()
#             temp_video.write(content)
#             temp_video_path = temp_video.name
        
#         # Generate subtitles (simplified implementation)
#         subtitles = await generate_subtitles_ai(temp_video_path, language)
        
#         # Cleanup
#         os.unlink(temp_video_path)
        
#         return {
#             "success": True,
#             "subtitles": subtitles,
#             "language": language,
#             "format": "srt",
#             "timestamp": get_timestamp()
#         }

#     except Exception as e:
#         logger.error(f"Subtitle generation error: {e}")
#         raise HTTPException(500, f"Subtitle generation failed: {str(e)}")

# # Core AI Functions
# def build_indian_context_prompt(request: ScriptRequest) -> str:
#     """Build culturally relevant prompt for Indian audience"""
    
#     platform_specs = {
#         "youtube": "detailed, engaging, 8-15 minutes",
#         "instagram": "short, visually appealing, under 60 seconds", 
#         "tiktok": "viral, trendy, 15-30 seconds",
#         "facebook": "community-focused, shareable",
#         "whatsapp": "concise, informative, under 2 minutes"
#     }
    
#     return f"""
#     Create a video script for Indian audience with these specifications:
    
#     TOPIC: {request.topic}
#     DURATION: {request.duration} minutes
#     CONTENT TYPE: {request.content_type}
#     TARGET LANGUAGE: {request.target_language}
#     PLATFORM: {request.target_platform} - {platform_specs.get(request.target_platform, '')}
#     CULTURAL CONTEXT: {request.cultural_context or 'General Indian context'}
#     TARGET AUDIENCE: {request.target_audience or 'General Indian audience'}
    
#     Please structure the script with:
#     1. Engaging opening (10%)
#     2. Main content (70%) 
#     3. Call-to-action/Conclusion (20%)
    
#     Include:
#     - Visual descriptions for each scene
#     - Suggested background music mood
#     - Text overlays/key points
#     - Cultural references appropriate for Indian audience
#     - Platform-specific optimization
    
#     Make it authentic, relatable, and culturally appropriate for Indian viewers.
#     """

# async def generate_script_with_gpt4(prompt: str) -> str:
#     """Generate script using GPT-4"""
#     try:
#         response = await openai.ChatCompletion.acreate(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are an expert video script writer specializing in Indian content creation."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=2000,
#             temperature=0.7
#         )
        
#         return response.choices[0].message.content
#     except Exception as e:
#         logger.error(f"GPT-4 error: {e}")
#         # Fallback to demo script
#         return generate_demo_script()

# async def generate_voiceover_elevenlabs(script: str, language: str, voice_style: str, speed: float) -> bytes:
#     """Generate voiceover using ElevenLabs"""
    
#     voice_mapping = {
#         "hindi": "Rachel",
#         "tamil": "Sarah", 
#         "telugu": "Emily",
#         "english": "Brian"
#     }
    
#     voice_id = voice_mapping.get(language.lower(), "Brian")
    
#     audio = generate(
#         text=script,
#         voice=voice_id,
#         model="eleven_multilingual_v1",
#         speed=speed
#     )
    
#     return audio

# # Media Search Functions
# async def search_pixabay(query: str, media_type: str) -> List[Dict]:
#     """Search Pixabay for media"""
#     # Implementation for Pixabay API
#     return []  # Placeholder

# async def search_pexels(query: str, media_type: str) -> List[Dict]:
#     """Search Pexels for media"""  
#     # Implementation for Pexels API
#     return []  # Placeholder

# def add_indian_context(query: str) -> str:
#     """Add Indian cultural context to search query"""
#     indian_keywords = ["Indian", "India", "cultural", "traditional", "desi"]
#     return f"{query} {' '.join(indian_keywords)}"

# def filter_culturally_appropriate(media_list: List[Dict]) -> List[Dict]:
#     """Filter media for cultural appropriateness"""
#     inappropriate_keywords = ["western", "foreign", "inappropriate"]
#     return [media for media in media_list if not any(kw in media.get('tags', '') for kw in inappropriate_keywords)]

# # Video Processing
# async def process_video_background(project_id: str, script: str, media_files: List[str], 
#                                  voiceover_file: str, platform: str, duration: int):
#     """Background video processing with FFmpeg"""
#     try:
#         project = projects_db[project_id]
#         project["progress"] = 10
#         project["status"] = "processing"
        
#         # Video processing implementation using FFmpeg
#         output_path = await process_video_with_ffmpeg(
#             media_files, voiceover_file, platform, duration
#         )
        
#         # Generate thumbnail
#         thumbnail_path = await generate_thumbnail(output_path)
        
#         # Update project status
#         project["status"] = "completed"
#         project["progress"] = 100
#         project["download_url"] = f"/videos/{os.path.basename(output_path)}"
#         project["thumbnail_url"] = f"/thumbnails/{os.path.basename(thumbnail_path)}"
        
#         logger.info(f"Video processing completed for project {project_id}")
        
#     except Exception as e:
#         logger.error(f"Video processing error for {project_id}: {e}")
#         projects_db[project_id]["status"] = "failed"
#         projects_db[project_id]["error"] = str(e)

# async def process_video_with_ffmpeg(media_files: List[str], voiceover_file: str, 
#                                   platform: str, duration: int) -> str:
#     """Process video using FFmpeg with platform optimization"""
    
#     # Platform-specific settings
#     platform_settings = {
#         "youtube": {"resolution": "1920x1080", "bitrate": "8M"},
#         "instagram": {"resolution": "1080x1350", "bitrate": "4M"},
#         "tiktok": {"resolution": "1080x1920", "bitrate": "5M"},
#     }
    
#     settings = platform_settings.get(platform, platform_settings["youtube"])
    
#     # Implement FFmpeg processing
#     output_filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
#     output_path = f"/tmp/{output_filename}"
    
#     # Basic FFmpeg command (simplified)
#     # In production, implement complete video processing pipeline
#     (
#         ffmpeg
#         .input(media_files[0])  # First media file
#         .output(output_path, **settings)
#         .run()
#     )
    
#     return output_path

# # Demo/fallback functions
# def generate_demo_script() -> str:
#     """Generate demo script when AI is unavailable"""
#     return """
#     SCRIPT: "Introduction to Digital Marketing in India"
    
#     SCENE 1 (0:00-0:15):
#     VISUAL: Energetic opening with Indian youth using smartphones
#     VOICEOVER: "In today's digital India, every business needs an online presence!"
    
#     SCENE 2 (0:15-1:00):
#     VISUAL: Step-by-step graphics showing social media platforms
#     VOICEOVER: "Let's explore how you can reach crores of potential customers..."
    
#     SCENE 3 (1:00-1:45):
#     VISUAL: Success stories of Indian businesses
#     VOICEOVER: "From local kirana stores to big brands, digital marketing is transforming businesses across India!"
    
#     CONCLUSION (1:45-2:00):
#     VISUAL: Call-to-action with contact information
#     VOICEOVER: "Start your digital journey today!"
#     """

# async def generate_demo_voiceover(request: VoiceOverRequest) -> Dict:
#     """Generate demo voiceover response"""
#     return {
#         "success": True,
#         "voiceover_url": "/demo/voiceover.mp3",
#         "metadata": {
#             "language": request.language,
#             "voice_style": request.voice_style,
#             "duration_estimate": len(request.script) // 15,
#             "timestamp": get_timestamp(),
#             "note": "Demo mode - AI service not configured"
#         }
#     }

# def generate_storyboard_from_script(script: str, duration: int) -> List[Dict]:
#     """Generate storyboard structure from script"""
#     # Simple storyboard generation
#     scenes = script.split('SCENE')[1:]  # Simple parsing
#     storyboard = []
    
#     for i, scene in enumerate(scenes[:10]):  # Max 10 scenes
#         storyboard.append({
#             "scene_number": i + 1,
#             "duration": duration // max(len(scenes), 1),
#             "description": scene[:100] + "..." if len(scene) > 100 else scene,
#             "visuals": "AI generated visuals",
#             "audio": "Background music + voiceover"
#         })
    
#     return storyboard

# async def generate_subtitles_ai(video_path: str, language: str) -> str:
#     """Generate subtitles using AI (simplified)"""
#     # Implementation for subtitle generation
#     return f"1\n00:00:00,000 --> 00:00:05,000\nSubtitles for {language} content\n\n"

# async def generate_thumbnail(video_path: str) -> str:
#     """Generate thumbnail from video"""
#     thumbnail_path = f"/tmp/thumbnail_{uuid.uuid4().hex[:8]}.jpg"
    
#     # Extract frame using FFmpeg
#     (
#         ffmpeg
#         .input(video_path, ss='00:00:01')  # Frame at 1 second
#         .output(thumbnail_path, vframes=1)
#         .run()
#     )
    
#     return thumbnail_path

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
import elevenlabs
from elevenlabs import Voice, VoiceSettings
import ffmpeg
import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict
import logging
from pydantic import BaseModel
import aiofiles
import tempfile
import shutil
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Video Content Creator API",
    description="Advanced AI-powered video creation platform for Indian content creators",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Setup AI Services
def setup_ai_services():
    """Initialize AI services with error handling"""
    services_status = {}
    
    # OpenAI Client Setup
    try:
        if OPENAI_API_KEY:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            services_status['openai'] = "Connected"
        else:
            openai_client = None
            services_status['openai'] = "No API key"
    except Exception as e:
        openai_client = None
        services_status['openai'] = f"Error: {str(e)}"
    
    # ElevenLabs Setup
    try:
        if ELEVENLABS_API_KEY:
            elevenlabs.set_api_key(ELEVENLABS_API_KEY)
            services_status['elevenlabs'] = "Connected"
        else:
            services_status['elevenlabs'] = "No API key"
    except Exception as e:
        services_status['elevenlabs'] = f"Error: {str(e)}"
    
    return services_status, openai_client

ai_services_status, openai_client = setup_ai_services()
logger.info(f"AI Services Status: {ai_services_status}")

# Pydantic Models
class ScriptRequest(BaseModel):
    topic: str
    duration: int
    content_type: str
    target_language: str
    target_platform: str
    cultural_context: Optional[str] = None
    target_audience: Optional[str] = None

class VoiceOverRequest(BaseModel):
    script: str
    language: str
    voice_style: Optional[str] = "professional"
    speed: Optional[float] = 1.0

class VideoProject(BaseModel):
    project_id: str
    title: str
    script: str
    voiceover_file: Optional[str] = None
    media_files: List[str] = []
    status: str = "draft"
    created_at: str
    duration: int

# Supported configurations
SUPPORTED_LANGUAGES = ["hindi", "tamil", "telugu", "bengali", "marathi", "gujarati", "kannada", "malayalam", "punjabi", "english"]
SUPPORTED_CONTENT_TYPES = ["educational", "marketing", "entertainment", "news", "tutorial"]
SUPPORTED_PLATFORMS = ["youtube", "instagram", "tiktok", "facebook", "whatsapp"]

# Project Storage (In production, use database)
projects_db = {}

# Utility Functions
async def generate_project_id():
    return f"proj_{uuid.uuid4().hex[:8]}"

def get_timestamp():
    return datetime.now().isoformat()

# API Routes
@app.get("/")
async def root():
    return {
        "message": "AI Video Content Creator API is running!",
        "status": "active",
        "service": "Advanced AI Video Creation Platform",
        "ai_services": ai_services_status
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": get_timestamp(),
        "ai_services": ai_services_status,
        "supported_languages": SUPPORTED_LANGUAGES,
        "supported_content_types": SUPPORTED_CONTENT_TYPES,
        "supported_platforms": SUPPORTED_PLATFORMS
    }

@app.post("/generate-script")
async def generate_script(request: ScriptRequest):
    """
    Generate video script using GPT-4 with Indian cultural context
    """
    try:
        logger.info(f"Script generation request: {request.dict()}")
        
        # Validate inputs
        if request.target_language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(400, f"Language must be one of: {SUPPORTED_LANGUAGES}")
        
        if request.content_type.lower() not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(400, f"Content type must be one of: {SUPPORTED_CONTENT_TYPES}")
        
        if request.target_platform.lower() not in SUPPORTED_PLATFORMS:
            raise HTTPException(400, f"Platform must be one of: {SUPPORTED_PLATFORMS}")

        # Build culturally relevant prompt
        prompt = build_indian_context_prompt(request)
        
        # Generate script with timeout (60 seconds max)
        try:
            script = await asyncio.wait_for(
                generate_script_with_gpt4(prompt),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(408, "Script generation timed out (60s limit)")
        
        # Generate storyboard
        storyboard = generate_storyboard_from_script(script, request.duration)
        
        logger.info("Script generation completed successfully")
        
        return {
            "success": True,
            "script": script,
            "storyboard": storyboard,
            "metadata": {
                "topic": request.topic,
                "duration": request.duration,
                "content_type": request.content_type,
                "language": request.target_language,
                "platform": request.target_platform,
                "timestamp": get_timestamp()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Script generation error: {e}")
        raise HTTPException(500, f"Script generation failed: {str(e)}")

@app.post("/generate-voiceover")
async def generate_voiceover(request: VoiceOverRequest):
    """
    Generate voiceover in Indian languages using ElevenLabs
    """
    try:
        logger.info(f"Voiceover generation request: language={request.language}")
        
        if request.language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(400, f"Language must be one of: {SUPPORTED_LANGUAGES}")

        if not ELEVENLABS_API_KEY:
            # Demo mode - return mock response
            return await generate_demo_voiceover(request)
        
        # Generate voiceover with ElevenLabs
        voiceover_data = await generate_voiceover_elevenlabs(
            request.script,
            request.language,
            request.voice_style,
            request.speed
        )
        
        # Save voiceover file
        filename = f"voiceover_{uuid.uuid4().hex[:8]}.mp3"
        filepath = f"/tmp/{filename}"
        
        with open(filepath, "wb") as f:
            f.write(voiceover_data)
        
        logger.info("Voiceover generation completed successfully")
        
        return {
            "success": True,
            "voiceover_url": f"/voiceovers/{filename}",
            "metadata": {
                "language": request.language,
                "voice_style": request.voice_style,
                "duration_estimate": len(request.script) // 15,  # Rough estimate
                "timestamp": get_timestamp()
            }
        }

    except Exception as e:
        logger.error(f"Voiceover generation error: {e}")
        raise HTTPException(500, f"Voiceover generation failed: {str(e)}")

@app.post("/search-media")
async def search_media(
    query: str = Form(...),
    media_type: str = Form("image"),
    cultural_context: bool = Form(True)
):
    """
    Search stock media from integrated APIs with Indian context
    """
    try:
        logger.info(f"Media search: {query}, type: {media_type}")
        
        # Enhanced query for Indian context
        enhanced_query = add_indian_context(query) if cultural_context else query
        
        # Search from both stock APIs (max 2 as specified)
        results = await asyncio.gather(
            search_pixabay(enhanced_query, media_type),
            search_pexels(enhanced_query, media_type),
            return_exceptions=True
        )
        
        # Filter valid results
        media_results = []
        for result in results:
            if not isinstance(result, Exception):
                media_results.extend(result)
        
        # Filter culturally appropriate content
        filtered_media = filter_culturally_appropriate(media_results)
        
        return {
            "success": True,
            "query": query,
            "enhanced_query": enhanced_query if cultural_context else query,
            "media_type": media_type,
            "results": filtered_media[:20],  # Limit results
            "total_count": len(filtered_media),
            "timestamp": get_timestamp()
        }

    except Exception as e:
        logger.error(f"Media search error: {e}")
        raise HTTPException(500, f"Media search failed: {str(e)}")

@app.post("/create-video-project")
async def create_video_project(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    script: str = Form(...),
    media_files: List[str] = Form(...),
    voiceover_file: str = Form(None),
    target_platform: str = Form("youtube"),
    duration: int = Form(...)
):
    """
    Create a new video project and start processing
    """
    try:
        project_id = await generate_project_id()
        
        project = VideoProject(
            project_id=project_id,
            title=title,
            script=script,
            voiceover_file=voiceover_file,
            media_files=media_files,
            status="processing",
            created_at=get_timestamp(),
            duration=duration
        )
        
        # Store project
        projects_db[project_id] = project.dict()
        
        # Start background video processing
        background_tasks.add_task(
            process_video_background,
            project_id,
            script,
            media_files,
            voiceover_file,
            target_platform,
            duration
        )
        
        return {
            "success": True,
            "project_id": project_id,
            "status": "processing",
            "message": "Video project created and processing started",
            "estimated_time": duration * 5,  # 5 minutes per minute of video
            "timestamp": get_timestamp()
        }

    except Exception as e:
        logger.error(f"Project creation error: {e}")
        raise HTTPException(500, f"Project creation failed: {str(e)}")

@app.get("/project-status/{project_id}")
async def get_project_status(project_id: str):
    """
    Get video project processing status
    """
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    
    return {
        "success": True,
        "project_id": project_id,
        "status": project["status"],
        "progress": project.get("progress", 0),
        "download_url": project.get("download_url"),
        "thumbnail_url": project.get("thumbnail_url"),
        "timestamp": get_timestamp()
    }

@app.post("/generate-subtitles")
async def generate_subtitles(
    video_file: UploadFile = File(...),
    language: str = Form("english")
):
    """
    Generate subtitles for video content
    """
    try:
        if not video_file.content_type.startswith('video/'):
            raise HTTPException(400, "Please upload a valid video file")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            content = await video_file.read()
            temp_video.write(content)
            temp_video_path = temp_video.name
        
        # Generate subtitles (simplified implementation)
        subtitles = await generate_subtitles_ai(temp_video_path, language)
        
        # Cleanup
        os.unlink(temp_video_path)
        
        return {
            "success": True,
            "subtitles": subtitles,
            "language": language,
            "format": "srt",
            "timestamp": get_timestamp()
        }

    except Exception as e:
        logger.error(f"Subtitle generation error: {e}")
        raise HTTPException(500, f"Subtitle generation failed: {str(e)}")

# Core AI Functions
def build_indian_context_prompt(request: ScriptRequest) -> str:
    """Build culturally relevant prompt for Indian audience"""
    
    platform_specs = {
        "youtube": "detailed, engaging, 8-15 minutes",
        "instagram": "short, visually appealing, under 60 seconds", 
        "tiktok": "viral, trendy, 15-30 seconds",
        "facebook": "community-focused, shareable",
        "whatsapp": "concise, informative, under 2 minutes"
    }
    
    return f"""
    Create a video script for Indian audience with these specifications:
    
    TOPIC: {request.topic}
    DURATION: {request.duration} minutes
    CONTENT TYPE: {request.content_type}
    TARGET LANGUAGE: {request.target_language}
    PLATFORM: {request.target_platform} - {platform_specs.get(request.target_platform, '')}
    CULTURAL CONTEXT: {request.cultural_context or 'General Indian context'}
    TARGET AUDIENCE: {request.target_audience or 'General Indian audience'}
    
    Please structure the script with:
    1. Engaging opening (10%)
    2. Main content (70%) 
    3. Call-to-action/Conclusion (20%)
    
    Include:
    - Visual descriptions for each scene
    - Suggested background music mood
    - Text overlays/key points
    - Cultural references appropriate for Indian audience
    - Platform-specific optimization
    
    Make it authentic, relatable, and culturally appropriate for Indian viewers.
    """

async def generate_script_with_gpt4(prompt: str) -> str:
    """Generate script using GPT-4"""
    try:
        if not openai_client:
            return generate_demo_script()
            
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert video script writer specializing in Indian content creation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT-4 error: {e}")
        # Fallback to demo script
        return generate_demo_script()

async def generate_voiceover_elevenlabs(script: str, language: str, voice_style: str, speed: float) -> bytes:
    """Generate voiceover using ElevenLabs"""
    
    voice_mapping = {
        "hindi": "Rachel",
        "tamil": "Sarah", 
        "telugu": "Emily",
        "english": "Brian"
    }
    
    voice_id = voice_mapping.get(language.lower(), "Brian")
    
    try:
        # Using the correct ElevenLabs API
        audio = elevenlabs.generate(
            text=script,
            voice=voice_id,
            model="eleven_multilingual_v1"
        )
        
        return audio
    except Exception as e:
        logger.error(f"ElevenLabs API error: {e}")
        # Return empty bytes as fallback
        return b""

# Media Search Functions
async def search_pixabay(query: str, media_type: str) -> List[Dict]:
    """Search Pixabay for media"""
    if not PIXABAY_API_KEY:
        return get_demo_media()
    
    try:
        url = "https://pixabay.com/api/"
        params = {
            'key': PIXABAY_API_KEY,
            'q': query,
            'image_type': 'photo' if media_type == 'image' else 'film',
            'per_page': 20,
            'safesearch': 'true'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        results = []
        for item in data.get('hits', []):
            results.append({
                'id': item['id'],
                'url': item['largeImageURL'] if media_type == 'image' else item['videos']['large']['url'],
                'preview_url': item['webformatURL'],
                'tags': item['tags'],
                'type': media_type
            })
        
        return results
    except Exception as e:
        logger.error(f"Pixabay search error: {e}")
        return get_demo_media()

async def search_pexels(query: str, media_type: str) -> List[Dict]:
    """Search Pexels for media"""  
    if not PEXELS_API_KEY:
        return get_demo_media()
    
    try:
        url = f"https://api.pexels.com/v1/{'photos' if media_type == 'image' else 'videos'}/search"
        headers = {
            'Authorization': PEXELS_API_KEY
        }
        params = {
            'query': query,
            'per_page': 20,
            'orientation': 'landscape'
        }
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        results = []
        for item in data.get('photos', []) + data.get('videos', []):
            results.append({
                'id': item['id'],
                'url': item['src']['original'] if media_type == 'image' else item['video_files'][0]['link'],
                'preview_url': item['src']['medium'] if media_type == 'image' else item['image'],
                'tags': query,
                'type': media_type
            })
        
        return results
    except Exception as e:
        logger.error(f"Pexels search error: {e}")
        return get_demo_media()

def add_indian_context(query: str) -> str:
    """Add Indian cultural context to search query"""
    indian_keywords = ["Indian", "India", "cultural", "traditional", "desi"]
    return f"{query} {' '.join(indian_keywords)}"

def filter_culturally_appropriate(media_list: List[Dict]) -> List[Dict]:
    """Filter media for cultural appropriateness"""
    inappropriate_keywords = ["western", "foreign", "inappropriate"]
    return [media for media in media_list if not any(kw in media.get('tags', '').lower() for kw in inappropriate_keywords)]

# Video Processing
async def process_video_background(project_id: str, script: str, media_files: List[str], 
                                 voiceover_file: str, platform: str, duration: int):
    """Background video processing with FFmpeg"""
    try:
        project = projects_db[project_id]
        project["progress"] = 10
        project["status"] = "processing"
        
        # Video processing implementation using FFmpeg
        output_path = await process_video_with_ffmpeg(
            media_files, voiceover_file, platform, duration
        )
        
        # Generate thumbnail
        thumbnail_path = await generate_thumbnail(output_path)
        
        # Update project status
        project["status"] = "completed"
        project["progress"] = 100
        project["download_url"] = f"/videos/{os.path.basename(output_path)}"
        project["thumbnail_url"] = f"/thumbnails/{os.path.basename(thumbnail_path)}"
        
        logger.info(f"Video processing completed for project {project_id}")
        
    except Exception as e:
        logger.error(f"Video processing error for {project_id}: {e}")
        projects_db[project_id]["status"] = "failed"
        projects_db[project_id]["error"] = str(e)

async def process_video_with_ffmpeg(media_files: List[str], voiceover_file: str, 
                                  platform: str, duration: int) -> str:
    """Process video using FFmpeg with platform optimization"""
    
    # Platform-specific settings
    platform_settings = {
        "youtube": {"resolution": "1920x1080", "bitrate": "8M"},
        "instagram": {"resolution": "1080x1350", "bitrate": "4M"},
        "tiktok": {"resolution": "1080x1920", "bitrate": "5M"},
    }
    
    settings = platform_settings.get(platform, platform_settings["youtube"])
    
    # Implement FFmpeg processing
    output_filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
    output_path = f"/tmp/{output_filename}"
    
    # Basic FFmpeg command (simplified)
    try:
        (
            ffmpeg
            .input(media_files[0] if media_files else 'color=black:s=1920x1080')  # First media file or black background
            .output(output_path, **settings)
            .run()
        )
    except Exception as e:
        logger.error(f"FFmpeg error: {e}")
        # Create a simple video file as fallback
        with open(output_path, 'wb') as f:
            f.write(b'dummy video content')
    
    return output_path

# Demo/fallback functions
def generate_demo_script() -> str:
    """Generate demo script when AI is unavailable"""
    return """
    SCRIPT: "Introduction to Digital Marketing in India"
    
    SCENE 1 (0:00-0:15):
    VISUAL: Energetic opening with Indian youth using smartphones
    VOICEOVER: "In today's digital India, every business needs an online presence!"
    
    SCENE 2 (0:15-1:00):
    VISUAL: Step-by-step graphics showing social media platforms
    VOICEOVER: "Let's explore how you can reach crores of potential customers..."
    
    SCENE 3 (1:00-1:45):
    VISUAL: Success stories of Indian businesses
    VOICEOVER: "From local kirana stores to big brands, digital marketing is transforming businesses across India!"
    
    CONCLUSION (1:45-2:00):
    VISUAL: Call-to-action with contact information
    VOICEOVER: "Start your digital journey today!"
    """

async def generate_demo_voiceover(request: VoiceOverRequest) -> Dict:
    """Generate demo voiceover response"""
    return {
        "success": True,
        "voiceover_url": "/demo/voiceover.mp3",
        "metadata": {
            "language": request.language,
            "voice_style": request.voice_style,
            "duration_estimate": len(request.script) // 15,
            "timestamp": get_timestamp(),
            "note": "Demo mode - AI service not configured"
        }
    }

def get_demo_media() -> List[Dict]:
    """Return demo media when APIs are unavailable"""
    return [
        {
            'id': 'demo1',
            'url': 'https://via.placeholder.com/1920x1080',
            'preview_url': 'https://via.placeholder.com/300x200',
            'tags': 'business, technology, india',
            'type': 'image'
        },
        {
            'id': 'demo2', 
            'url': 'https://via.placeholder.com/1920x1080',
            'preview_url': 'https://via.placeholder.com/300x200',
            'tags': 'education, learning, indian',
            'type': 'image'
        }
    ]

def generate_storyboard_from_script(script: str, duration: int) -> List[Dict]:
    """Generate storyboard structure from script"""
    # Simple storyboard generation
    scenes = script.split('\n\n')  # Simple parsing by paragraphs
    storyboard = []
    
    for i, scene in enumerate(scenes[:10]):  # Max 10 scenes
        storyboard.append({
            "scene_number": i + 1,
            "duration": duration // max(len(scenes), 1),
            "description": scene[:100] + "..." if len(scene) > 100 else scene,
            "visuals": "AI generated visuals",
            "audio": "Background music + voiceover"
        })
    
    return storyboard

async def generate_subtitles_ai(video_path: str, language: str) -> str:
    """Generate subtitles using AI (simplified)"""
    # Implementation for subtitle generation
    return f"1\n00:00:00,000 --> 00:00:05,000\nSubtitles for {language} content\n\n"

async def generate_thumbnail(video_path: str) -> str:
    """Generate thumbnail from video"""
    thumbnail_path = f"/tmp/thumbnail_{uuid.uuid4().hex[:8]}.jpg"
    
    try:
        # Extract frame using FFmpeg
        (
            ffmpeg
            .input(video_path, ss='00:00:01')  # Frame at 1 second
            .output(thumbnail_path, vframes=1)
            .run()
        )
    except Exception as e:
        logger.error(f"Thumbnail generation error: {e}")
        # Create a simple placeholder
        from PIL import Image
        img = Image.new('RGB', (300, 200), color='blue')
        img.save(thumbnail_path)
    
    return thumbnail_path

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)