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

# from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from openai import OpenAI
# import elevenlabs
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
# import requests

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
    
#     # OpenAI Client Setup
#     try:
#         if OPENAI_API_KEY:
#             openai_client = OpenAI(api_key=OPENAI_API_KEY)
#             services_status['openai'] = "Connected"
#         else:
#             openai_client = None
#             services_status['openai'] = "No API key"
#     except Exception as e:
#         openai_client = None
#         services_status['openai'] = f"Error: {str(e)}"
    
#     # ElevenLabs Setup
#     try:
#         if ELEVENLABS_API_KEY:
#             elevenlabs.set_api_key(ELEVENLABS_API_KEY)
#             services_status['elevenlabs'] = "Connected"
#         else:
#             services_status['elevenlabs'] = "No API key"
#     except Exception as e:
#         services_status['elevenlabs'] = f"Error: {str(e)}"
    
#     return services_status, openai_client

# ai_services_status, openai_client = setup_ai_services()
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
#                 "duration_estimate": len(request.script) // 15,
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
#             "results": filtered_media[:20],
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
#         projects_db[project_id] = project.model_dump()
        
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
#             "estimated_time": duration * 5,
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
#         if not openai_client:
#             return generate_demo_script()
            
#         response = openai_client.chat.completions.create(
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
    
#     try:
#         audio = elevenlabs.generate(
#             text=script,
#             voice=voice_id,
#             model="eleven_multilingual_v1"
#         )
#         return audio
#     except Exception as e:
#         logger.error(f"ElevenLabs API error: {e}")
#         return b""

# # Media Search Functions
# async def search_pixabay(query: str, media_type: str) -> List[Dict]:
#     """Search Pixabay for media"""
#     if not PIXABAY_API_KEY:
#         return get_demo_media()
    
#     try:
#         url = "https://pixabay.com/api/"
#         params = {
#             'key': PIXABAY_API_KEY,
#             'q': query,
#             'image_type': 'photo' if media_type == 'image' else 'film',
#             'per_page': 10,
#             'safesearch': 'true'
#         }
        
#         response = requests.get(url, params=params)
#         data = response.json()
        
#         results = []
#         for item in data.get('hits', []):
#             results.append({
#                 'id': item['id'],
#                 'url': item['largeImageURL'] if media_type == 'image' else item['videos']['large']['url'],
#                 'preview_url': item['webformatURL'],
#                 'tags': item['tags'],
#                 'type': media_type
#             })
        
#         return results
#     except Exception as e:
#         logger.error(f"Pixabay search error: {e}")
#         return get_demo_media()

# async def search_pexels(query: str, media_type: str) -> List[Dict]:
#     """Search Pexels for media"""  
#     if not PEXELS_API_KEY:
#         return get_demo_media()
    
#     try:
#         url = f"https://api.pexels.com/v1/{'photos' if media_type == 'image' else 'videos'}/search"
#         headers = {'Authorization': PEXELS_API_KEY}
#         params = {'query': query, 'per_page': 10, 'orientation': 'landscape'}
        
#         response = requests.get(url, headers=headers, params=params)
#         data = response.json()
        
#         results = []
#         for item in data.get('photos', []) + data.get('videos', []):
#             results.append({
#                 'id': item['id'],
#                 'url': item['src']['original'] if media_type == 'image' else item['video_files'][0]['link'],
#                 'preview_url': item['src']['medium'] if media_type == 'image' else item['image'],
#                 'tags': query,
#                 'type': media_type
#             })
        
#         return results
#     except Exception as e:
#         logger.error(f"Pexels search error: {e}")
#         return get_demo_media()

# def add_indian_context(query: str) -> str:
#     """Add Indian cultural context to search query"""
#     indian_keywords = ["Indian", "India", "cultural", "traditional", "desi"]
#     return f"{query} {' '.join(indian_keywords)}"

# def filter_culturally_appropriate(media_list: List[Dict]) -> List[Dict]:
#     """Filter media for cultural appropriateness"""
#     inappropriate_keywords = ["western", "foreign", "inappropriate"]
#     return [media for media in media_list if not any(kw in media.get('tags', '').lower() for kw in inappropriate_keywords)]

# # Video Processing
# async def process_video_background(project_id: str, script: str, media_files: List[str], 
#                                  voiceover_file: str, platform: str, duration: int):
#     """Background video processing with FFmpeg"""
#     try:
#         project = projects_db[project_id]
#         project["progress"] = 10
#         project["status"] = "processing"
        
#         # Simple processing simulation
#         await asyncio.sleep(2)  # Simulate processing time
        
#         # Update project status
#         project["status"] = "completed"
#         project["progress"] = 100
#         project["download_url"] = f"/videos/demo_{project_id}.mp4"
#         project["thumbnail_url"] = f"/thumbnails/demo_{project_id}.jpg"
        
#         logger.info(f"Video processing completed for project {project_id}")
        
#     except Exception as e:
#         logger.error(f"Video processing error for {project_id}: {e}")
#         projects_db[project_id]["status"] = "failed"
#         projects_db[project_id]["error"] = str(e)

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

# def get_demo_media() -> List[Dict]:
#     """Return demo media when APIs are unavailable"""
#     return [
#         {
#             'id': 'demo1',
#             'url': 'https://via.placeholder.com/1920x1080',
#             'preview_url': 'https://via.placeholder.com/300x200',
#             'tags': 'business, technology, india',
#             'type': 'image'
#         },
#         {
#             'id': 'demo2', 
#             'url': 'https://via.placeholder.com/1920x1080',
#             'preview_url': 'https://via.placeholder.com/300x200',
#             'tags': 'education, learning, indian',
#             'type': 'image'
#         }
#     ]

# def generate_storyboard_from_script(script: str, duration: int) -> List[Dict]:
#     """Generate storyboard structure from script"""
#     scenes = script.split('\n\n')
#     storyboard = []
    
#     for i, scene in enumerate(scenes[:5]):
#         storyboard.append({
#             "scene_number": i + 1,
#             "duration": duration // max(len(scenes), 1),
#             "description": scene[:100] + "..." if len(scene) > 100 else scene,
#             "visuals": "AI generated visuals",
#             "audio": "Background music + voiceover"
#         })
    
#     return storyboard

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import elevenlabs
from elevenlabs import set_api_key, generate
import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict
import logging
from pydantic import BaseModel
import tempfile
import shutil
import requests
from pathlib import Path
import time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import aiofiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Video Content Creator API",
    description="Advanced AI-powered video creation platform for Indian content creators",
    version="2.0.0"
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# Create directories
Path("static/voiceovers").mkdir(parents=True, exist_ok=True)
Path("static/videos").mkdir(parents=True, exist_ok=True)
Path("static/thumbnails").mkdir(parents=True, exist_ok=True)
Path("static/temp").mkdir(parents=True, exist_ok=True)

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
            set_api_key(ELEVENLABS_API_KEY)
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
    progress: int = 0
    download_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

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

def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """Clean up files older than specified hours"""
    try:
        current_time = time.time()
        for file_path in directory.glob("*"):
            if file_path.is_file() and (current_time - file_path.stat().st_mtime) > (max_age_hours * 3600):
                file_path.unlink()
                logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")

# API Routes
@app.get("/")
@limiter.limit("100/minute")
async def root(request: Request):
    return {
        "message": "AI Video Content Creator API is running!",
        "status": "active",
        "service": "Advanced AI Video Creation Platform",
        "ai_services": ai_services_status,
        "version": "2.0.0"
    }

@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    # Clean up old files on health check
    cleanup_old_files(Path("static/temp"))
    cleanup_old_files(Path("static/voiceovers"))
    cleanup_old_files(Path("static/videos"))
    
    return {
        "status": "healthy",
        "timestamp": get_timestamp(),
        "ai_services": ai_services_status,
        "supported_languages": SUPPORTED_LANGUAGES,
        "supported_content_types": SUPPORTED_CONTENT_TYPES,
        "supported_platforms": SUPPORTED_PLATFORMS,
        "storage": {
            "voiceovers": len(list(Path("static/voiceovers").glob("*.mp3"))),
            "videos": len(list(Path("static/videos").glob("*.mp4"))),
            "thumbnails": len(list(Path("static/thumbnails").glob("*.jpg")))
        }
    }

@app.post("/generate-script")
@limiter.limit("30/minute")
async def generate_script(request: Request, script_request: ScriptRequest):
    """
    Generate video script using GPT-4 with Indian cultural context
    """
    try:
        logger.info(f"Script generation request from {request.client.host}: {script_request.dict()}")
        
        # Validate inputs
        if script_request.target_language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(400, f"Language must be one of: {SUPPORTED_LANGUAGES}")
        
        if script_request.content_type.lower() not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(400, f"Content type must be one of: {SUPPORTED_CONTENT_TYPES}")
        
        if script_request.target_platform.lower() not in SUPPORTED_PLATFORMS:
            raise HTTPException(400, f"Platform must be one of: {SUPPORTED_PLATFORMS}")

        # Build culturally relevant prompt
        prompt = build_indian_context_prompt(script_request)
        
        # Generate script with timeout (60 seconds max)
        try:
            script = await asyncio.wait_for(
                generate_script_with_gpt4(prompt),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"Script generation timeout for {request.client.host}")
            raise HTTPException(408, "Script generation timed out (60s limit)")
        
        # Generate storyboard
        storyboard = generate_storyboard_from_script(script, script_request.duration)
        
        logger.info(f"Script generation completed successfully for {request.client.host}")
        
        return {
            "success": True,
            "script": script,
            "storyboard": storyboard,
            "metadata": {
                "topic": script_request.topic,
                "duration": script_request.duration,
                "content_type": script_request.content_type,
                "language": script_request.target_language,
                "platform": script_request.target_platform,
                "timestamp": get_timestamp()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Script generation error for {request.client.host}: {e}")
        raise HTTPException(500, f"Script generation failed: {str(e)}")

@app.post("/generate-voiceover")
@limiter.limit("20/minute")
async def generate_voiceover(request: Request, voiceover_request: VoiceOverRequest):
    """
    Generate voiceover in Indian languages using ElevenLabs
    """
    try:
        logger.info(f"Voiceover generation request from {request.client.host}: language={voiceover_request.language}")
        
        if voiceover_request.language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(400, f"Language must be one of: {SUPPORTED_LANGUAGES}")

        if not ELEVENLABS_API_KEY:
            # Demo mode - return mock response
            return await generate_demo_voiceover(voiceover_request)
        
        # Generate voiceover with ElevenLabs
        voiceover_filename = await generate_voiceover_elevenlabs(
            voiceover_request.script,
            voiceover_request.language,
            voiceover_request.voice_style,
            voiceover_request.speed
        )
        
        logger.info(f"Voiceover generation completed successfully for {request.client.host}")
        
        return {
            "success": True,
            "voiceover_url": f"/static/voiceovers/{voiceover_filename}",
            "metadata": {
                "language": voiceover_request.language,
                "voice_style": voiceover_request.voice_style,
                "duration_estimate": len(voiceover_request.script) // 15,
                "timestamp": get_timestamp()
            }
        }

    except Exception as e:
        logger.error(f"Voiceover generation error for {request.client.host}: {e}")
        raise HTTPException(500, f"Voiceover generation failed: {str(e)}")

@app.post("/search-media")
@limiter.limit("40/minute")
async def search_media(
    request: Request,
    query: str = Form(...),
    media_type: str = Form("image"),
    cultural_context: bool = Form(True)
):
    """
    Search stock media from integrated APIs with Indian context
    """
    try:
        logger.info(f"Media search from {request.client.host}: {query}, type: {media_type}")
        
        # Enhanced query for Indian context
        enhanced_query = add_indian_context(query) if cultural_context else query
        
        # Search from both stock APIs
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
            "results": filtered_media[:20],
            "total_count": len(filtered_media),
            "timestamp": get_timestamp()
        }

    except Exception as e:
        logger.error(f"Media search error for {request.client.host}: {e}")
        raise HTTPException(500, f"Media search failed: {str(e)}")

@app.post("/create-video-project")
@limiter.limit("10/minute")
async def create_video_project(
    request: Request,
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    script: str = Form(...),
    media_files: str = Form("[]"),
    voiceover_file: str = Form(None),
    target_platform: str = Form("youtube"),
    duration: int = Form(...)
):
    """
    Create a new video project and start processing
    """
    try:
        logger.info(f"Creating video project from {request.client.host}: {title}")
        
        project_id = await generate_project_id()
        
        # Parse media_files from JSON string
        try:
            media_files_list = json.loads(media_files)
        except Exception as e:
            logger.warning(f"Failed to parse media_files, using empty list: {e}")
            media_files_list = []
        
        project = VideoProject(
            project_id=project_id,
            title=title,
            script=script,
            voiceover_file=voiceover_file,
            media_files=media_files_list,
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
            media_files_list,
            voiceover_file,
            target_platform,
            duration
        )
        
        logger.info(f"Video project created successfully: {project_id}")
        
        return {
            "success": True,
            "project_id": project_id,
            "status": "processing",
            "message": "Video project created and processing started",
            "estimated_time": duration * 10,  # 10 seconds per minute of video
            "timestamp": get_timestamp()
        }

    except Exception as e:
        logger.error(f"Project creation error for {request.client.host}: {e}")
        raise HTTPException(500, f"Project creation failed: {str(e)}")

@app.get("/project-status/{project_id}")
@limiter.limit("60/minute")
async def get_project_status(request: Request, project_id: str):
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

@app.get("/static/voiceovers/{filename}")
async def get_voiceover(filename: str):
    """Serve voiceover files"""
    file_path = Path("static/voiceovers") / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
    raise HTTPException(404, "Voiceover file not found")

@app.get("/static/videos/{filename}")
async def get_video(filename: str):
    """Serve video files"""
    file_path = Path("static/videos") / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    raise HTTPException(404, "Video file not found")

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
    Return the script in a clear, structured format with scene descriptions.
    """

async def generate_script_with_gpt4(prompt: str) -> str:
    """Generate script using GPT-4 with fallback"""
    try:
        if not openai_client:
            logger.info("Using demo script - OpenAI client not available")
            return generate_demo_script()
        
        # Try different models with fallback
        models_to_try = ["gpt-4", "gpt-3.5-turbo"]
        
        for model in models_to_try:
            try:
                logger.info(f"Attempting to generate script with {model}")
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert video script writer specializing in Indian content creation. Create engaging, culturally appropriate scripts."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.7
                )
                
                script = response.choices[0].message.content
                logger.info(f"Successfully generated script with {model}")
                return script
                
            except Exception as model_error:
                logger.warning(f"Model {model} failed: {model_error}")
                continue
        
        # If all models fail, use demo script
        logger.info("All AI models failed, using demo script")
        return generate_demo_script()
        
    except Exception as e:
        logger.error(f"GPT script generation error: {e}")
        return generate_demo_script()

async def generate_voiceover_elevenlabs(script: str, language: str, voice_style: str, speed: float) -> str:
    """Generate voiceover using ElevenLabs"""
    
    voice_mapping = {
        "hindi": "Rachel",
        "tamil": "Sarah", 
        "telugu": "Emily",
        "bengali": "Charlotte",
        "marathi": "Charlotte",
        "gujarati": "Charlotte", 
        "kannada": "Charlotte",
        "malayalam": "Charlotte",
        "punjabi": "Charlotte",
        "english": "Brian"
    }
    
    voice_id = voice_mapping.get(language.lower(), "Brian")
    
    try:
        logger.info(f"Generating voiceover in {language} with voice {voice_id}")
        
        audio = generate(
            text=script,
            voice=voice_id,
            model="eleven_multilingual_v1"
        )
        
        # Save the file properly
        filename = f"voiceover_{uuid.uuid4().hex[:8]}.mp3"
        filepath = Path("static/voiceovers") / filename
        
        with open(filepath, "wb") as f:
            f.write(audio)
        
        return filename
        
    except Exception as e:
        logger.error(f"ElevenLabs API error: {e}")
        # Create a demo voiceover file
        return await create_demo_voiceover_file()

async def create_demo_voiceover_file() -> str:
    """Create a demo voiceover file when ElevenLabs fails"""
    filename = f"demo_voiceover_{uuid.uuid4().hex[:8]}.mp3"
    filepath = Path("static/voiceovers") / filename
    
    # Create an empty demo file
    with open(filepath, "wb") as f:
        f.write(b"Demo voiceover content")
    
    return filename

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
            'per_page': 10,
            'safesearch': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = []
        for item in data.get('hits', []):
            results.append({
                'id': str(item['id']),
                'url': item['largeImageURL'] if media_type == 'image' else item.get('videos', {}).get('large', {}).get('url', ''),
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
        headers = {'Authorization': PEXELS_API_KEY}
        params = {
            'query': query, 
            'per_page': 10, 
            'orientation': 'landscape'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        
        results = []
        items = data.get('photos', []) if media_type == 'image' else data.get('videos', [])
        
        for item in items:
            if media_type == 'image':
                results.append({
                    'id': str(item['id']),
                    'url': item['src']['original'],
                    'preview_url': item['src']['medium'],
                    'tags': query,
                    'type': media_type
                })
            else:
                video_files = item.get('video_files', [])
                if video_files:
                    results.append({
                        'id': str(item['id']),
                        'url': video_files[0]['link'],
                        'preview_url': item['image'],
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
    inappropriate_keywords = ["western", "foreign", "inappropriate", "sensitive"]
    filtered_media = []
    
    for media in media_list:
        tags = media.get('tags', '').lower()
        if not any(kw in tags for kw in inappropriate_keywords):
            filtered_media.append(media)
    
    return filtered_media

# Video Processing with MoviePy
async def process_video_background(project_id: str, script: str, media_files: List[str], 
                                 voiceover_file: str, platform: str, duration: int):
    """Background video processing with MoviePy"""
    try:
        project = projects_db[project_id]
        project["progress"] = 10
        project["status"] = "processing"
        
        logger.info(f"Starting video processing for project {project_id}")
        
        # Download media files
        project["progress"] = 20
        downloaded_media = await download_media_files(media_files)
        
        # Get voiceover file path
        voiceover_path = None
        if voiceover_file and voiceover_file.startswith('/static/voiceovers/'):
            voiceover_filename = voiceover_file.split('/')[-1]
            voiceover_path = Path("static/voiceovers") / voiceover_filename
        
        project["progress"] = 40
        
        # Create video using MoviePy
        video_filename = f"{project_id}.mp4"
        video_path = Path("static/videos") / video_filename
        
        await create_video_with_moviepy(
            downloaded_media,
            voiceover_path,
            video_path,
            duration,
            project
        )
        
        # Create thumbnail
        thumbnail_filename = f"{project_id}.jpg"
        thumbnail_path = Path("static/thumbnails") / thumbnail_filename
        await create_thumbnail(video_path, thumbnail_path)
        
        # Update project status
        project["status"] = "completed"
        project["progress"] = 100
        project["download_url"] = f"/static/videos/{video_filename}"
        project["thumbnail_url"] = f"/static/thumbnails/{thumbnail_filename}"
        
        # Cleanup temporary files
        for media_path in downloaded_media:
            if Path(media_path).exists():
                Path(media_path).unlink()
        
        logger.info(f"Video processing completed for project {project_id}")
        
    except Exception as e:
        logger.error(f"Video processing error for {project_id}: {str(e)}")
        projects_db[project_id]["status"] = "failed"
        projects_db[project_id]["error"] = str(e)

async def download_media_files(media_urls: List[str]) -> List[str]:
    """Download media files to temporary storage"""
    downloaded_paths = []
    
    for i, url in enumerate(media_urls):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                filename = f"temp_media_{uuid.uuid4().hex[:8]}.jpg"
                filepath = Path("static/temp") / filename
                
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                downloaded_paths.append(str(filepath))
                logger.info(f"Downloaded media {i+1}/{len(media_urls)}")
            else:
                logger.warning(f"Failed to download media: {url}")
        except Exception as e:
            logger.error(f"Error downloading media {url}: {e}")
    
    return downloaded_paths

async def create_video_with_moviepy(media_paths: List[str], voiceover_path: str, 
                                  output_path: Path, duration: int, project: dict):
    """Create video using MoviePy with proper compositing"""
    try:
        project["progress"] = 50
        
        # Calculate duration per image
        total_duration = duration * 60  # Convert to seconds
        clips = []
        
        if media_paths:
            duration_per_image = total_duration / len(media_paths)
            
            for i, media_path in enumerate(media_paths):
                try:
                    # Create image clip with calculated duration
                    clip = ImageClip(media_path, duration=duration_per_image)
                    
                    # Resize to 1920x1080 (Full HD)
                    clip = clip.resize(height=1080)
                    
                    # Center the image
                    clip = clip.set_position('center')
                    
                    clips.append(clip)
                    
                    # Update progress
                    progress = 50 + (i / len(media_paths)) * 30
                    project["progress"] = int(progress)
                    
                except Exception as e:
                    logger.error(f"Error processing image {media_path}: {e}")
                    continue
        
        # If no media available, create a simple color clip
        if not clips:
            from moviepy.video.VideoClip import ColorClip
            clip = ColorClip(size=(1920, 1080), color=(100, 100, 200), duration=total_duration)
            clips = [clip]
        
        project["progress"] = 80
        
        # Create final video clip
        final_clip = CompositeVideoClip(clips, size=(1920, 1080))
        
        # Add voiceover if available
        if voiceover_path and Path(voiceover_path).exists():
            try:
                audio_clip = AudioFileClip(str(voiceover_path))
                # Adjust audio duration to match video
                if audio_clip.duration > total_duration:
                    audio_clip = audio_clip.subclip(0, total_duration)
                final_clip = final_clip.set_audio(audio_clip)
            except Exception as e:
                logger.error(f"Error adding audio: {e}")
        
        project["progress"] = 90
        
        # Write video file
        final_clip.write_videofile(
            str(output_path),
            fps=24,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # Close clips to free memory
        final_clip.close()
        for clip in clips:
            clip.close()
        
    except Exception as e:
        logger.error(f"MoviePy video creation error: {e}")
        raise

async def create_thumbnail(video_path: Path, thumbnail_path: Path):
    """Create thumbnail from video"""
    try:
        if video_path.exists():
            clip = VideoFileClip(str(video_path))
            # Get frame at 10% of video duration
            thumbnail_time = clip.duration * 0.1
            clip.save_frame(str(thumbnail_path), t=thumbnail_time)
            clip.close()
    except Exception as e:
        logger.error(f"Thumbnail creation error: {e}")

# Demo/fallback functions
def generate_demo_script() -> str:
    """Generate demo script when AI is unavailable"""
    return """VIDEO SCRIPT: "Introduction to Digital Marketing in India"

SCENE 1 (0:00-0:15):
VISUAL: Dynamic opening with Indian youth using smartphones, colorful graphics
VOICEOVER: "Namaste! In today's digital India, every business needs a strong online presence to reach millions of potential customers!"

SCENE 2 (0:15-0:45):
VISUAL: Step-by-step animation showing social media platforms popular in India
VOICEOVER: "From Facebook and Instagram to WhatsApp Business, learn how to connect with your audience where they spend their time..."

SCENE 3 (0:45-1:30):
VISUAL: Success stories showing Indian businesses - local kirana store to startup
VOICEOVER: "Meet Raju Bhaiya's grocery store that doubled sales using simple WhatsApp marketing. Or Priya's fashion boutique that reached customers across India through Instagram..."

SCENE 4 (1:30-2:00):
VISUAL: Call-to-action screen with contact information and next steps
VOICEOVER: "Ready to grow your business? Start your digital journey today! Follow these simple steps and watch your business transform."

CONCLUSION:
• Use social media consistently
• Engage with your customers
• Track your results
• Keep learning and adapting

END SCREEN:
"Digital India - Digital You"
Contact: www.example.com | Phone: +91-XXXXXX-XXXX
"""

async def generate_demo_voiceover(request: VoiceOverRequest) -> Dict:
    """Generate demo voiceover response"""
    # Create a demo voiceover file
    filename = await create_demo_voiceover_file()
    
    return {
        "success": True,
        "voiceover_url": f"/static/voiceovers/{filename}",
        "metadata": {
            "language": request.language,
            "voice_style": request.voice_style,
            "duration_estimate": len(request.script) // 15,
            "timestamp": get_timestamp(),
            "note": "Demo mode - ElevenLabs service not configured"
        }
    }

def get_demo_media() -> List[Dict]:
    """Return demo media when APIs are unavailable"""
    return [
        {
            'id': 'demo1',
            'url': 'https://images.unsplash.com/photo-1556740758-90de205c2d1b?w=500',
            'preview_url': 'https://images.unsplash.com/photo-1556740758-90de205c2d1b?w=300',
            'tags': 'business, technology, india, startup',
            'type': 'image'
        },
        {
            'id': 'demo2', 
            'url': 'https://images.unsplash.com/photo-1552664730-d307ca884978?w=500',
            'preview_url': 'https://images.unsplash.com/photo-1552664730-d307ca884978?w=300',
            'tags': 'education, learning, indian students',
            'type': 'image'
        },
        {
            'id': 'demo3',
            'url': 'https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=500',
            'preview_url': 'https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=300',
            'tags': 'marketing, digital, social media',
            'type': 'image'
        }
    ]

def generate_storyboard_from_script(script: str, duration: int) -> List[Dict]:
    """Generate storyboard structure from script"""
    lines = script.split('\n')
    scenes = []
    current_scene = []
    
    for line in lines:
        if line.strip().startswith('SCENE') or line.strip().startswith('VISUAL:') or line.strip().startswith('END'):
            if current_scene:
                scenes.append('\n'.join(current_scene))
                current_scene = []
        current_scene.append(line)
    
    if current_scene:
        scenes.append('\n'.join(current_scene))
    
    storyboard = []
    scene_duration = duration // max(len(scenes), 1)
    
    for i, scene in enumerate(scenes[:8]):  # Max 8 scenes
        storyboard.append({
            "scene_number": i + 1,
            "duration": scene_duration,
            "description": scene[:150] + "..." if len(scene) > 150 else scene,
            "visuals": "AI generated visuals matching scene content",
            "audio": "Background music + professional voiceover",
            "transition": "Smooth cross-fade"
        })
    
    return storyboard

# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)