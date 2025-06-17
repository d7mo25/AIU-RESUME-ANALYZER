import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Form
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, EmailStr
    from typing import Optional, List, Dict, Any
    
    import base64
    import io
    import re
    import time
    import jwt
    import bcrypt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    # PDF processing
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    
    # Firebase (optional)
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage
        FIREBASE_AVAILABLE = True
        logger.info("✅ Firebase imports successful")
    except ImportError as e:
        FIREBASE_AVAILABLE = False
        logger.warning(f"⚠️ Firebase not available: {e}")
    
    # spaCy (optional)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("✅ spaCy model loaded")
    except Exception as e:
        nlp = None
        SPACY_AVAILABLE = False
        logger.warning(f"⚠️ spaCy not available: {e}")
    
    logger.info("✅ All imports successful")
    
except ImportError as e:
    logger.error(f"❌ Critical import error: {e}")
    sys.exit(1)

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "railway-production-secret-key-change-this")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
BUCKET_NAME = os.getenv("FIREBASE_STORAGE_BUCKET", "resume-analyzer-d58fd.appspot.com")

# Global variables
firebase_initialized = False
db = None
bucket = None
IN_MEMORY_USERS = {}
IN_MEMORY_RESUMES = {}
IN_MEMORY_ADMINS = {}

def initialize_firebase():
    """Initialize Firebase with graceful error handling"""
    global firebase_initialized, db, bucket
    
    if not FIREBASE_AVAILABLE:
        logger.warning("⚠️ Firebase not available, skipping initialization")
        return False
    
    try:
        # Check if already initialized
        try:
            firebase_admin.get_app()
            firebase_initialized = True
            db = firestore.client()
            bucket = storage.bucket(BUCKET_NAME)
            logger.info("✅ Firebase already initialized")
            return True
        except ValueError:
            pass
        
        # Try to initialize
        firebase_config_json = os.getenv('FIREBASE_SERVICE_ACCOUNT')
        if firebase_config_json:
            service_account_info = json.loads(firebase_config_json)
            cred = credentials.Certificate(service_account_info)
            logger.info("✅ Using Firebase service account from environment")
        else:
            # Try local file
            if Path('serviceAccountKey.json').exists():
                cred = credentials.Certificate('serviceAccountKey.json')
                logger.info("✅ Using Firebase service account from file")
            else:
                raise Exception("No Firebase credentials found")
        
        firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
        db = firestore.client()
        bucket = storage.bucket(BUCKET_NAME)
        firebase_initialized = True
        
        # Create default admin
        create_default_admin()
        logger.info("✅ Firebase initialized successfully")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Firebase initialization failed: {e}")
        firebase_initialized = False
        return False

def create_default_admin():
    """Create default admin account"""
    if not db:
        return
    
    try:
        admins_ref = db.collection('admins')
        query = admins_ref.where('email', '==', "admin@aiu.edu.my").limit(1).get()
        
        if not query:
            hashed_password = bcrypt.hashpw("Admin123!".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            admin_doc = {
                "email": "admin@aiu.edu.my",
                "password_hash": hashed_password,
                "created_at": datetime.now(),
                "last_login": None,
                "created_by": "system"
            }
            admins_ref.add(admin_doc)
            logger.info("✅ Created default admin")
    except Exception as e:
        logger.error(f"❌ Error creating default admin: {e}")

def setup_fallback_storage():
    """Setup in-memory storage as fallback"""
    global IN_MEMORY_ADMINS
    
    logger.info("📝 Setting up in-memory storage")
    
    # Create default admin for in-memory storage
    hashed_password = bcrypt.hashpw("Admin123!".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    IN_MEMORY_ADMINS["default_admin"] = {
        "email": "admin@aiu.edu.my",
        "password_hash": hashed_password,
        "created_at": datetime.now(),
        "last_login": None
    }
    logger.info("✅ In-memory storage ready")

# Pydantic models
class UserRegistration(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    confirm_password: str
    phone: str
    agree_terms: bool

class UserLogin(BaseModel):
    email: str
    password: str
    rememberMe: bool = False

class AdminLogin(BaseModel):
    email: str
    password: str

class DashboardStats(BaseModel):
    total_users: int
    total_resumes: int
    average_score: float

# Keyword categories for analysis
KEYWORD_CATEGORIES = {
    "Technical Skills": [
        "LMS", "Moodle", "Blackboard", "management", "Excel", "Microsoft Office",
        "SPSS", "Statistical software", "Research methodology", "Teaching tools"
    ],
    "Soft Skills": ["Communication", "Leadership", "Teamwork", "Adaptability", "Problem-solving"],
    "Work Experience": [
        "University lecturer", "Professor role", "Course development", "Teaching assistant",
        "Administrative experience", "Project management", "Fresh graduate", "Internship"
    ],
    "Language Proficiency": ["English"],
    "Achievements": ["Research grants", "Employee recognition", "Certifications"],
    "Publications": ["Academic articles", "Research papers", "Conference proceedings"],
    "Candidate Profile": ["Full Name", "Email", "Phone", "Address", "LinkedIn"]
}

MAX_SCORES = {
    "Technical Skills": 10, "Soft Skills": 15, "Work Experience": 10,
    "Language Proficiency": 10, "Achievements": 5, "Publications": 10,
    "Sections Presence": 25, "Candidate Profile": 15
}

# Utility functions
def create_jwt_token(user_data: dict) -> str:
    payload = {
        "user_id": user_data.get("user_id"),
        "email": user_data.get("email"),
        "role": user_data.get("role", "user"),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def pdf_reader(file_buffer):
    """Extract text from PDF"""
    try:
        resource_manager = PDFResourceManager()
        output_string = io.StringIO()
        laparams = LAParams(char_margin=2.0, line_margin=0.5, boxes_flow=0.5)
        converter = TextConverter(resource_manager, output_string, laparams=laparams)
        interpreter = PDFPageInterpreter(resource_manager, converter)
        file_buffer.seek(0)
        
        for page in PDFPage.get_pages(file_buffer, caching=True, check_extractable=True, maxpages=0):
            interpreter.process_page(page)
                
        converter.close()
        text = output_string.getvalue()
        output_string.close()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_basic_info_from_text(text):
    """Extract basic information from resume text"""
    # Extract name using spaCy if available
    name = "N/A"
    if SPACY_AVAILABLE and nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                    name = ent.text.strip()
                    break
        except Exception:
            pass
    
    # Email extraction
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    
    # Phone extraction
    phone_pattern = re.compile(r'(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|\+?6?0?1?[-.\s]?[0-9]{1,2}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{4})')
    phone_matches = list(phone_pattern.finditer(text))
    phone = phone_matches[0].group(0).strip() if phone_matches else "N/A"
    
    return {
        'name': name,
        'email': email_match.group(0) if email_match else "N/A",
        'phone': phone,
        'address': "N/A",
        'linkedin': "N/A",
        'github': "N/A"
    }

# Initialize FastAPI app
def create_app():
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="AIU Smart Resume Analyzer",
        version="1.0.0",
        docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")
        logger.info("✅ Static files mounted")
    
    # Setup templates
    templates_dir = Path("templates")
    templates = None
    if templates_dir.exists():
        templates = Jinja2Templates(directory="templates")
        logger.info("✅ Templates initialized")
    
    security = HTTPBearer()
    
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        return verify_jwt_token(credentials.credentials)
    
    # Routes
    @app.get("/")
    async def root():
        return {
            "message": "AIU Resume Analyzer is running!",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "firebase": firebase_initialized,
            "spacy": SPACY_AVAILABLE,
            "version": "1.0.0"
        }
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "firebase": firebase_initialized,
                "spacy": SPACY_AVAILABLE,
                "templates": templates is not None
            }
        }
    
    @app.get("/ping")
    async def ping():
        return {"ping": "pong"}
    
    # Page routes
    if templates:
        @app.get("/login", response_class=HTMLResponse)
        @app.get("/index", response_class=HTMLResponse) 
        async def login_page(request: Request):
            return templates.TemplateResponse("index.html", {"request": request})
        
        @app.get("/register", response_class=HTMLResponse)
        async def register_page(request: Request):
            return templates.TemplateResponse("register.html", {"request": request})
        
        @app.get("/admin-login", response_class=HTMLResponse)
        async def admin_login_page(request: Request):
            return templates.TemplateResponse("admin-login.html", {"request": request})
        
        @app.get("/admin-dashboard", response_class=HTMLResponse)
        async def admin_dashboard_page(request: Request):
            return templates.TemplateResponse("admin-dashboard.html", {"request": request})
        
        @app.get("/resume-upload", response_class=HTMLResponse)
        async def resume_upload_page(request: Request):
            return templates.TemplateResponse("resume-upload.html", {"request": request})
    
    # Authentication routes
    @app.post("/api/login")
    async def user_login(user_data: UserLogin):
        logger.info(f"🔐 Login attempt: {user_data.email}")
        
        try:
            if db:
                users_ref = db.collection('users')
                docs = users_ref.where('email', '==', user_data.email).limit(1).get()
                
                if len(docs) == 0:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                user_doc = docs[0].to_dict()
                user_id = docs[0].id
            else:
                # Use in-memory storage
                user_found = None
                for uid, user in IN_MEMORY_USERS.items():
                    if user['email'] == user_data.email:
                        user_found = user
                        user_id = uid
                        break
                
                if not user_found:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                user_doc = user_found
            
            if not bcrypt.checkpw(user_data.password.encode('utf-8'), user_doc['password_hash'].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            user_info = {"user_id": user_id, "email": user_doc['email'], "role": "user"}
            token = create_jwt_token(user_info)
            
            return {"message": "Login successful", "token": token, "user": user_info}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            raise HTTPException(status_code=500, detail="Login failed")
    
    @app.post("/api/admin/login")
    async def admin_login(admin_data: AdminLogin):
        logger.info(f"👑 Admin login attempt: {admin_data.email}")
        
        try:
            if db:
                admins_ref = db.collection('admins')
                docs = admins_ref.where('email', '==', admin_data.email).limit(1).get()
                
                if len(docs) == 0:
                    raise HTTPException(status_code=401, detail="Invalid admin credentials")
                
                admin_doc = docs[0].to_dict()
                admin_id = docs[0].id
            else:
                # Use in-memory storage
                admin_found = None
                for aid, admin in IN_MEMORY_ADMINS.items():
                    if admin['email'] == admin_data.email:
                        admin_found = admin
                        admin_id = aid
                        break
                
                if not admin_found:
                    raise HTTPException(status_code=401, detail="Invalid admin credentials")
                admin_doc = admin_found
            
            if not bcrypt.checkpw(admin_data.password.encode('utf-8'), admin_doc['password_hash'].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            
            admin_info = {"user_id": admin_id, "email": admin_data.email, "role": "admin"}
            token = create_jwt_token(admin_info)
            
            return {"message": "Admin login successful", "token": token, "user": admin_info}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Admin login error: {e}")
            raise HTTPException(status_code=500, detail="Admin login failed")
    
    @app.post("/api/register")
    async def register_user(user_data: UserRegistration):
        logger.info(f"👤 Registration attempt: {user_data.email}")
        
        if user_data.password != user_data.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        user_doc = {
            "full_name": user_data.full_name,
            "email": user_data.email,
            "phone": user_data.phone,
            "password_hash": hashed_password,
            "created_at": datetime.now(),
            "role": "user"
        }
        
        try:
            if db:
                users_ref = db.collection('users')
                # Check if user exists
                query = users_ref.where('email', '==', user_data.email).limit(1).stream()
                if any(query):
                    raise HTTPException(status_code=400, detail="Email already registered")
                
                doc_ref = users_ref.add(user_doc)
                user_id = doc_ref[1].id
            else:
                # Use in-memory storage
                if any(u['email'] == user_data.email for u in IN_MEMORY_USERS.values()):
                    raise HTTPException(status_code=400, detail="Email already registered")
                
                user_id = f"user_{len(IN_MEMORY_USERS) + 1}"
                IN_MEMORY_USERS[user_id] = user_doc
            
            user_info = {"user_id": user_id, "email": user_data.email, "role": "user"}
            token = create_jwt_token(user_info)
            
            return {"message": "Registration successful", "token": token, "user": user_info}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            raise HTTPException(status_code=500, detail="Registration failed")
    
    # Admin stats
    @app.get("/api/admin/stats", response_model=DashboardStats)
    async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        try:
            if db:
                users_ref = db.collection('users')
                resumes_ref = db.collection('resumes')
                
                total_users = len(list(users_ref.stream()))
                resume_docs = list(resumes_ref.stream())
                total_resumes = len(resume_docs)
                
                analyzed_resumes = [doc for doc in resume_docs if doc.to_dict().get('ats_score')]
                average_score = sum(doc.to_dict().get('ats_score', 0) for doc in analyzed_resumes) / len(analyzed_resumes) if analyzed_resumes else 0.0
            else:
                total_users = len(IN_MEMORY_USERS)
                total_resumes = len(IN_MEMORY_RESUMES)
                analyzed_resumes = [r for r in IN_MEMORY_RESUMES.values() if r.get('ats_score')]
                average_score = sum(r.get('ats_score', 0) for r in analyzed_resumes) / len(analyzed_resumes) if analyzed_resumes else 0.0
            
            return DashboardStats(
                total_users=total_users,
                total_resumes=total_resumes,
                average_score=round(average_score, 1)
            )
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch stats")
    
    return app

# Create the application
app = create_app()

# Initialize services
async def startup_services():
    """Initialize services on startup"""
    logger.info("🔧 Initializing services...")
    
    # Initialize Firebase
    firebase_success = initialize_firebase()
    
    # Setup fallback storage if Firebase fails
    if not firebase_success:
        setup_fallback_storage()
    
    logger.info("✅ Services initialized")

# Run startup
try:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(startup_services())
    loop.close()
except Exception as e:
    logger.warning(f"⚠️ Startup services error: {e}")
    setup_fallback_storage()

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    try:
        port = int(os.getenv("PORT", 8000))
        
        logger.info("🚀 Starting AIU Resume Analyzer")
        logger.info(f"📍 Port: {port}")
        logger.info(f"🔥 Firebase: {firebase_initialized}")
        logger.info(f"🧠 spaCy: {SPACY_AVAILABLE}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start: {e}")
        sys.exit(1)
