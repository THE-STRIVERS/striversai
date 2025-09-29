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
from elevenlabs import set_api_key, generate
import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict
import logging
from pydantic import BaseModel
import requests
from pathlib import Path
import time

# MoviePy imports
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, ColorClip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Video Content Creator API",
    description="AI-powered video creation platform",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Directories
BASE_DIR = Path("static")
VOICEOVER_DIR = BASE_DIR / "voiceovers"
VIDEO_DIR = BASE_DIR / "videos"
THUMBNAIL_DIR = BASE_DIR / "thumbnails"
TEMP_DIR = BASE_DIR / "temp"

for directory in [BASE_DIR, VOICEOVER_DIR, VIDEO_DIR, THUMBNAIL_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Setup AI Services - FIXED
def setup_ai_services():
    services_status = {}
    
    try:
        if OPENAI_API_KEY:
            # Fixed: Simple initialization without extra params
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            services_status['openai'] = "Connected"
        else:
            openai_client = None
            services_status['openai'] = "No API key"
    except Exception as e:
        openai_client = None
        services_status['openai'] = f"Error: {str(e)}"
        logger.error(f"OpenAI setup error: {e}")
    
    try:
        if ELEVENLABS_API_KEY:
            set_api_key(ELEVENLABS_API_KEY)
            services_status['elevenlabs'] = "Connected"
        else:
            services_status['elevenlabs'] = "No API key"
    except Exception as e:
        services_status['elevenlabs'] = f"Error: {str(e)}"
        logger.error(f"ElevenLabs setup error: {e}")
    
    return services_status, openai_client

ai_services_status, openai_client = setup_ai_services()
logger.info(f"AI Services Status: {ai_services_status}")

# Models
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

# Constants
SUPPORTED_LANGUAGES = ["hindi", "tamil", "telugu", "bengali", "marathi", "gujarati", "kannada", "malayalam", "punjabi", "english"]
SUPPORTED_CONTENT_TYPES = ["educational", "marketing", "entertainment", "news", "tutorial"]
SUPPORTED_PLATFORMS = ["youtube", "instagram", "tiktok", "facebook", "whatsapp"]

projects_db = {}

# Utils
async def generate_project_id():
    return f"proj_{uuid.uuid4().hex[:8]}"

def get_timestamp():
    return datetime.now().isoformat()

def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    try:
        current_time = time.time()
        for file_path in directory.glob("*"):
            if file_path.is_file():
                if (current_time - file_path.stat().st_mtime) > (max_age_hours * 3600):
                    file_path.unlink()
                    logger.info(f"Cleaned up: {file_path}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# Routes
@app.get("/")
async def root():
    return {
        "message": "AI Video Creator API",
        "status": "active",
        "version": "2.0.0",
        "ai_services": ai_services_status
    }

@app.get("/health")
async def health_check():
    cleanup_old_files(TEMP_DIR, 2)
    cleanup_old_files(VOICEOVER_DIR, 48)
    cleanup_old_files(VIDEO_DIR, 48)
    
    return {
        "status": "healthy",
        "timestamp": get_timestamp(),
        "ai_services": ai_services_status
    }

@app.post("/generate-script")
async def generate_script(script_request: ScriptRequest):
    try:
        logger.info(f"Script request: {script_request.topic}")
        
        if script_request.target_language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(400, "Unsupported language")
        
        prompt = f"""Create a {script_request.duration} minute video script about: {script_request.topic}
Content Type: {script_request.content_type}
Language: {script_request.target_language}
Platform: {script_request.target_platform}

Structure with scenes, dialogue, and visual descriptions."""

        try:
            script = await asyncio.wait_for(
                generate_script_ai(prompt),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(408, "Timeout")
        
        storyboard = generate_storyboard(script, script_request.duration)
        
        return {
            "success": True,
            "script": script,
            "storyboard": storyboard,
            "metadata": {
                "topic": script_request.topic,
                "duration": script_request.duration,
                "timestamp": get_timestamp()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Script error: {e}")
        raise HTTPException(500, str(e))

@app.post("/generate-voiceover")
async def generate_voiceover(voiceover_request: VoiceOverRequest):
    try:
        logger.info(f"Voiceover: {voiceover_request.language}")
        
        if voiceover_request.language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(400, "Unsupported language")

        filename = await create_voiceover(
            voiceover_request.script,
            voiceover_request.language,
            voiceover_request.voice_style
        )
        
        return {
            "success": True,
            "voiceover_url": f"/static/voiceovers/{filename}",
            "metadata": {
                "language": voiceover_request.language,
                "timestamp": get_timestamp()
            }
        }

    except Exception as e:
        logger.error(f"Voiceover error: {e}")
        raise HTTPException(500, str(e))

@app.post("/search-media")
async def search_media(
    query: str = Form(...),
    media_type: str = Form("image"),
    cultural_context: bool = Form(True)
):
    try:
        logger.info(f"Media search: {query}")
        
        enhanced_query = f"{query} India Indian" if cultural_context else query
        
        results = await asyncio.gather(
            search_pixabay(enhanced_query, media_type),
            search_pexels(enhanced_query, media_type),
            return_exceptions=True
        )
        
        media_results = []
        for result in results:
            if not isinstance(result, Exception):
                media_results.extend(result)
        
        return {
            "success": True,
            "query": query,
            "results": media_results[:20],
            "timestamp": get_timestamp()
        }

    except Exception as e:
        logger.error(f"Media error: {e}")
        raise HTTPException(500, str(e))

@app.post("/create-video-project")
async def create_video_project(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    script: str = Form(...),
    media_files: str = Form("[]"),
    voiceover_file: str = Form(None),
    target_platform: str = Form("youtube"),
    duration: int = Form(...)
):
    try:
        logger.info(f"Creating: {title}")
        
        project_id = await generate_project_id()
        
        try:
            media_files_list = json.loads(media_files)
        except:
            media_files_list = []
        
        project = {
            "project_id": project_id,
            "title": title,
            "script": script,
            "voiceover_file": voiceover_file,
            "media_files": media_files_list,
            "status": "processing",
            "created_at": get_timestamp(),
            "duration": duration,
            "progress": 0
        }
        
        projects_db[project_id] = project
        
        background_tasks.add_task(
            process_video,
            project_id,
            media_files_list,
            voiceover_file,
            duration
        )
        
        return {
            "success": True,
            "project_id": project_id,
            "status": "processing",
            "timestamp": get_timestamp()
        }

    except Exception as e:
        logger.error(f"Project error: {e}")
        raise HTTPException(500, str(e))

@app.get("/project-status/{project_id}")
async def get_project_status(project_id: str):
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    
    return {
        "success": True,
        "project_id": project_id,
        "status": project["status"],
        "progress": project.get("progress", 0),
        "download_url": project.get("download_url"),
        "timestamp": get_timestamp()
    }

@app.get("/static/videos/{filename}")
async def download_video(filename: str):
    file_path = VIDEO_DIR / filename
    if file_path.exists():
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename=filename
        )
    raise HTTPException(404, "Video not found")

# AI Functions
async def generate_script_ai(prompt: str) -> str:
    try:
        if not openai_client:
            return get_demo_script()
        
        for model in ["gpt-4", "gpt-3.5-turbo"]:
            try:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert video script writer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"{model} failed: {e}")
                continue
        
        return get_demo_script()
        
    except Exception as e:
        logger.error(f"AI error: {e}")
        return get_demo_script()

async def create_voiceover(script: str, language: str, voice_style: str) -> str:
    voice_map = {"hindi": "Rachel", "english": "Brian"}
    voice_id = voice_map.get(language.lower(), "Rachel")
    
    try:
        if ELEVENLABS_API_KEY:
            audio = generate(
                text=script[:1000],
                voice=voice_id,
                model="eleven_multilingual_v1"
            )
            
            filename = f"voice_{uuid.uuid4().hex[:8]}.mp3"
            filepath = VOICEOVER_DIR / filename
            
            with open(filepath, "wb") as f:
                f.write(audio)
            
            return filename
    except Exception as e:
        logger.error(f"ElevenLabs error: {e}")
    
    filename = f"demo_{uuid.uuid4().hex[:8]}.mp3"
    filepath = VOICEOVER_DIR / filename
    with open(filepath, "wb") as f:
        f.write(b"Demo audio")
    return filename

async def search_pixabay(query: str, media_type: str) -> List[Dict]:
    if not PIXABAY_API_KEY:
        return get_demo_media()
    
    try:
        response = requests.get(
            "https://pixabay.com/api/",
            params={
                'key': PIXABAY_API_KEY,
                'q': query,
                'image_type': 'photo',
                'per_page': 10,
                'safesearch': 'true'
            },
            timeout=10
        )
        data = response.json()
        
        return [{
            'id': str(item['id']),
            'url': item['largeImageURL'],
            'preview_url': item['webformatURL'],
            'tags': item['tags'],
            'type': media_type
        } for item in data.get('hits', [])]
        
    except Exception as e:
        logger.error(f"Pixabay: {e}")
        return get_demo_media()

async def search_pexels(query: str, media_type: str) -> List[Dict]:
    if not PEXELS_API_KEY:
        return get_demo_media()
    
    try:
        response = requests.get(
            "https://api.pexels.com/v1/photos/search",
            headers={'Authorization': PEXELS_API_KEY},
            params={'query': query, 'per_page': 10},
            timeout=10
        )
        data = response.json()
        
        return [{
            'id': str(item['id']),
            'url': item['src']['original'],
            'preview_url': item['src']['medium'],
            'tags': query,
            'type': media_type
        } for item in data.get('photos', [])]
        
    except Exception as e:
        logger.error(f"Pexels: {e}")
        return get_demo_media()

# Video Processing
async def process_video(project_id: str, media_urls: List[str], 
                       voiceover_file: str, duration: int):
    try:
        project = projects_db[project_id]
        project["progress"] = 10
        
        logger.info(f"Processing: {project_id}")
        
        project["progress"] = 30
        media_paths = await download_media(media_urls)
        
        voiceover_path = None
        if voiceover_file and voiceover_file.startswith('/static/voiceovers/'):
            filename = voiceover_file.split('/')[-1]
            voiceover_path = VOICEOVER_DIR / filename
        
        project["progress"] = 50
        
        video_filename = f"{project_id}.mp4"
        video_path = VIDEO_DIR / video_filename
        
        await create_video_moviepy(
            media_paths,
            voiceover_path,
            video_path,
            duration,
            project
        )
        
        for path in media_paths:
            Path(path).unlink(missing_ok=True)
        
        project["status"] = "completed"
        project["progress"] = 100
        project["download_url"] = f"/static/videos/{video_filename}"
        
        logger.info(f"Completed: {project_id}")
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        projects_db[project_id]["status"] = "failed"
        projects_db[project_id]["error"] = str(e)

async def download_media(urls: List[str]) -> List[str]:
    paths = []
    for i, url in enumerate(urls[:5]):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
                filepath = TEMP_DIR / filename
                
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                paths.append(str(filepath))
                logger.info(f"Downloaded {i+1}/{len(urls)}")
        except Exception as e:
            logger.error(f"Download error: {e}")
    
    return paths

async def create_video_moviepy(media_paths: List[str], voiceover_path: Path,
                               output_path: Path, duration: int, project: dict):
    try:
        project["progress"] = 60
        
        total_duration = duration * 60
        clips = []
        
        if media_paths:
            duration_per_image = total_duration / len(media_paths)
            
            for i, media_path in enumerate(media_paths):
                try:
                    clip = ImageClip(media_path, duration=duration_per_image)
                    clip = clip.resize(height=1080)
                    clips.append(clip)
                    
                    progress = 60 + (i / len(media_paths)) * 20
                    project["progress"] = int(progress)
                except Exception as e:
                    logger.error(f"Image error: {e}")
        
        if not clips:
            clip = ColorClip(size=(1920, 1080), color=(100, 100, 200), 
                           duration=total_duration)
            clips = [clip]
        
        project["progress"] = 80
        
        final_clip = concatenate_videoclips(clips, method="compose")
        
        if voiceover_path and voiceover_path.exists():
            try:
                audio = AudioFileClip(str(voiceover_path))
                if audio.duration > total_duration:
                    audio = audio.subclip(0, total_duration)
                final_clip = final_clip.set_audio(audio)
            except Exception as e:
                logger.error(f"Audio error: {e}")
        
        project["progress"] = 90
        
        final_clip.write_videofile(
            str(output_path),
            fps=24,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',
            verbose=False,
            logger=None
        )
        
        final_clip.close()
        for clip in clips:
            clip.close()
        
    except Exception as e:
        logger.error(f"MoviePy error: {e}")
        raise

# Demo
def get_demo_script() -> str:
    return """VIDEO SCRIPT: Digital Marketing

SCENE 1 (0:00-0:30):
Opening with digital devices
"Welcome to digital marketing!"

SCENE 2 (0:30-1:30):
Social media platforms
"Learn to connect with millions..."

SCENE 3 (1:30-2:00):
Call to action
"Start your journey today!"
"""

def get_demo_media() -> List[Dict]:
    return [{
        'id': f'demo{i}',
        'url': f'https://images.unsplash.com/photo-{1556740758 + i * 1000}?w=1920',
        'preview_url': f'https://images.unsplash.com/photo-{1556740758 + i * 1000}?w=300',
        'tags': 'business',
        'type': 'image'
    } for i in range(5)]

def generate_storyboard(script: str, duration: int) -> List[Dict]:
    return [{
        "scene_number": i + 1,
        "duration": duration // 3,
        "description": f"Scene {i + 1}",
        "visuals": "AI content"
    } for i in range(3)]

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)