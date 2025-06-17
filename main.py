from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
import base64
import pandas as pd
import io
import re
import spacy
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import time
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage, exceptions
import requests
import json
import random
from datetime import datetime, timedelta
import jwt
import os
from pathlib import Path
import tempfile
import matplotlib
import bcrypt
import urllib.request
import urllib.parse
import csv
import logging
import sys
matplotlib.use('Agg')

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy model (should be pre-installed in Docker)
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("✅ spaCy model loaded successfully")
except OSError as e:
    logger.error(f"❌ Failed to load spaCy model: {e}")
    # Fallback - this shouldn't happen in production Docker
    nlp = None

# Initialize FastAPI app
app = FastAPI(
    title="AIU Smart Resume Analyzer", 
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security and templates
security = HTTPBearer()

# Check if directories exist
static_dir = Path("static")
templates_dir = Path("templates")

if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted")
else:
    logger.warning("⚠️ Static directory not found")

if templates_dir.exists():
    templates = Jinja2Templates(directory="templates")
    logger.info("✅ Templates directory found")
else:
    logger.warning("⚠️ Templates directory not found")
    templates = None

# Configuration
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "AIzaSyBSBSkxJoya3yk4JA8wXp6BgF99GQJplrs")
BUCKET_NAME = os.getenv("FIREBASE_STORAGE_BUCKET", "resume-analyzer-d58fd.appspot.com")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secure-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Firebase initialization
firebase_initialized = False
db = None
bucket = None
firebase_app = None

def initialize_firebase():
    """Initialize Firebase with better error handling"""
    global firebase_initialized, db, bucket, firebase_app
    
    try:
        # Check if Firebase is already initialized
        firebase_app = firebase_admin.get_app()
        firebase_initialized = True
        db = firestore.client()
        bucket = storage.bucket(BUCKET_NAME)
        logger.info("✅ Firebase already initialized")
        return True
    except ValueError:
        pass
    
    try:
        # Try environment variable first
        firebase_config_json = os.getenv('FIREBASE_SERVICE_ACCOUNT')
        if firebase_config_json:
            service_account_info = json.loads(firebase_config_json)
            cred = credentials.Certificate(service_account_info)
            logger.info("✅ Using Firebase service account from environment")
        else:
            # Try local file
            service_account_files = ['serviceAccountKey.json', 'firebase-service-account.json']
            cred = None
            for file_path in service_account_files:
                if os.path.exists(file_path):
                    cred = credentials.Certificate(file_path)
                    logger.info(f"✅ Using Firebase service account from {file_path}")
                    break
            
            if not cred:
                raise Exception("No Firebase credentials found")
        
        firebase_app = firebase_admin.initialize_app(cred, {'storageBucket': BUCKET_NAME})
        db = firestore.client()
        bucket = storage.bucket(BUCKET_NAME)
        firebase_initialized = True
        
        # Create default admin
        create_default_admin()
        logger.info("✅ Firebase initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        firebase_initialized = False
        return False

def create_default_admin():
    """Create default admin if it doesn't exist"""
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
            logger.info(f"✅ Created default admin: admin@aiu.edu.my / Admin123!")
    except Exception as e:
        logger.error(f"❌ Error creating default admin: {str(e)}")

# Initialize Firebase (non-blocking)
try:
    firebase_status = initialize_firebase()
except Exception as e:
    logger.error(f"❌ Firebase initialization error: {e}")
    firebase_status = False

# Fallback in-memory storage for development/testing
if not firebase_initialized:
    logger.warning("⚠️ Using in-memory storage (Firebase unavailable)")
    IN_MEMORY_ADMIN = {
        "email": "admin@aiu.edu.my",
        "password_hash": bcrypt.hashpw("Admin123!".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "created_at": datetime.now(),
        "last_login": None
    }
    IN_MEMORY_USERS = {}
    IN_MEMORY_RESUMES = {}
    IN_MEMORY_ADMINS = {"dev_admin": IN_MEMORY_ADMIN}

# Pydantic models
class TokenRequest(BaseModel):
    token: str

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

class AdminCreate(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str

class CheckEmailRequest(BaseModel):
    email: EmailStr

class DatabasePasswordUpdate(BaseModel):
    email: EmailStr
    newPassword: str

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    bio: Optional[str] = None

class UserProfileResponse(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    bio: Optional[str] = None
    created_at: str
    last_login: Optional[str] = None
    role: str

class DashboardStats(BaseModel):
    total_users: int
    total_resumes: int
    average_score: float

# Updated Keyword Categories
KEYWORD_CATEGORIES = {
    "Technical Skills": [
        "LMS", "Moodle", "Blackboard", "management",
        "E-learning platforms", "SPSS", "Statistical software", "Excel", "Microsoft Excel",
        "Microsoft Office", "Qualitative analysis", "Research methodology",
        "Teaching tools", "Academic software", "Learning analytics"
    ],
    "Soft Skills": [
        "Communication", "Leadership", "Teamwork", "Adaptability", "Problem-solving"
    ],
    "Work Experience": [
        "University lecturer", "Professor role",
        "Course development", "Lecture preparation",
        "Student mentorship", "Research supervision", "Teaching assistant",
        "Administrative experience", "Technical support", "Project management", "Office management",
        "Data analysis", "Systems administrator", "Technical documentation",
        "Operations management", "Fresh graduate", "Internship",
        "Industrial training"
    ],
    "Language Proficiency": ["English"],
    "Achievements": [
        "Research grants", "Employee recognition", "Dean's list", "Competition achievements",
        "Certifications"
    ],
    "Publications": [
        "Peer-reviewed journal articles", "Conference proceedings", "Books", "Blog",
        "Academic articles", "Research papers"
    ],
    "Candidate Profile": ["Full Name", "Email", "Phone", "Address", "LinkedIn"]
}

# Section patterns for detection
SECTION_PATTERNS = {
    "Candidate Profile": [
        r'\b(personal\s+information|contact\s+information|profile|about\s+me|summary)\b',
        r'\b(name|email|phone|address|linkedin)\b'
    ],
    "Education": [
        r'\b(education|academic\s+background|qualifications|degree|university|college|school)\b',
        r'\b(bachelor|master|phd|diploma|certificate|graduation)\b'
    ],
    "Skills": [
        r'\b(skills|competencies|technical\s+skills|core\s+competencies|abilities)\b',
        r'\b(programming|software|tools|languages|proficiency)\b'
    ],
    "Experience": [
        r'\b(experience|work\s+experience|employment|career|professional\s+experience)\b',
        r'\b(job|position|role|internship|worked\s+at|employed)\b'
    ]
}

REQUIRED_SECTIONS = ["Candidate Profile", "Education", "Skills", "Experience"]

# Maximum scores for each category
MAX_SCORES = {
    "Technical Skills": 10,
    "Soft Skills": 15,
    "Work Experience": 10,
    "Language Proficiency": 10,
    "Achievements": 5,
    "Publications": 10,
    "Sections Presence": 25,
    "Candidate Profile": 15
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
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return verify_jwt_token(credentials.credentials)

def pdf_reader(file_buffer):
    try:
        resource_manager = PDFResourceManager()
        output_string = io.StringIO()
        laparams = LAParams(char_margin=2.0, line_margin=0.5, boxes_flow=0.5, detect_vertical=True)
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

def extract_full_name(text):
    """Extract full name using spaCy NER"""
    if not nlp:
        return "N/A"
    
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
            return ent.text.strip()
    return "N/A"

def extract_basic_info_from_text(text):
    """Enhanced basic info extraction"""
    name = extract_full_name(text)
    
    # Email extraction
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    
    # Phone extraction
    phone_pattern = re.compile(r'(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|\+?6?0?1?[-.\s]?[0-9]{1,2}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{4})')
    phone_matches = list(phone_pattern.finditer(text))
    phone = "N/A"
    if phone_matches:
        earliest_match = min(phone_matches, key=lambda m: m.start())
        phone = earliest_match.group(0).strip()
    
    # LinkedIn extraction
    linkedin_match = re.search(r"(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    
    # GitHub extraction
    github_match = re.search(r"(https?://)?(www\.)?github\.com/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    
    return {
        'name': name,
        'email': email_match.group(0) if email_match else "N/A",
        'phone': phone,
        'address': "N/A",
        'linkedin': linkedin_match.group(0) if linkedin_match else "N/A",
        'github': github_match.group(0) if github_match else "N/A",
    }

def extract_keywords(text, keywords):
    """Extract keywords from resume text based on categories"""
    text_lower = text.lower()
    found = []
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(kw)
    return found

def extract_candidate_profile_keywords(basic_info):
    """Convert extracted basic information into keyword format for scoring"""
    profile_keywords = []

    if basic_info.get('name') and basic_info['name'] != "N/A":
        profile_keywords.append("Full Name")
    if basic_info.get('email') and basic_info['email'] != "N/A":
        profile_keywords.append("Email")
    if basic_info.get('phone') and basic_info['phone'] != "N/A":
        profile_keywords.append("Phone")
    if basic_info.get('address') and basic_info['address'] != "N/A":
        profile_keywords.append("Address")
    if basic_info.get('linkedin') and basic_info['linkedin'] != "N/A":
        profile_keywords.append("LinkedIn")

    return profile_keywords

def detect_sections_presence(text):
    """Detect the presence of key sections in the resume text"""
    text_lower = text.lower()
    found_sections = []

    for section, patterns in SECTION_PATTERNS.items():
        section_found = False
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                section_found = True
                break
        if section_found:
            found_sections.append(section)

    return found_sections

def calculate_ats_score(extraction_results, sections_found):
    """Calculate ATS score"""
    category_scores = {}
    for category, found_keywords in extraction_results.items():
        max_score = MAX_SCORES.get(category, 0)
        found_count = len(found_keywords)
        total_possible = len(KEYWORD_CATEGORIES.get(category, []))
        
        if total_possible > 0:
            category_scores[category] = round((found_count / total_possible) * max_score, 2)
        else:
            category_scores[category] = 0

    # Calculate sections presence score
    sections_score = (len(sections_found) / len(REQUIRED_SECTIONS)) * MAX_SCORES["Sections Presence"]
    category_scores["Sections Presence"] = round(sections_score, 2)

    # Total score is sum of all category scores
    total_score = sum(category_scores.values())
    
    return round(min(total_score, 100), 2), category_scores

def create_keyword_chart(extraction_results, sections_found, applied_role: str):
    """Create keyword analysis chart"""
    categories = list(extraction_results.keys()) + ["Sections Presence"]
    found_counts = [len(extraction_results[cat]) for cat in extraction_results.keys()] + [len(sections_found)]
    total_keywords = [len(KEYWORD_CATEGORIES[cat]) for cat in extraction_results.keys()] + [len(REQUIRED_SECTIONS)]
    
    _, category_scores = calculate_ats_score(extraction_results, sections_found)
    scores = [category_scores[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(categories))

    bars1 = ax.barh(y_pos, found_counts, color='#667eea', alpha=0.8, label='Keywords Found')
    bars2 = ax.barh(y_pos, total_keywords, color='none', edgecolor='gray', linewidth=1.5, label='Total Possible')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title(f'Resume Analysis Results for {applied_role}', fontsize=16, fontweight='bold')

    max_scores_list = [MAX_SCORES[cat] for cat in categories]
    for i, (score, max_score) in enumerate(zip(scores, max_scores_list)):
        ax.text(max(found_counts[i], 1) + 0.3, i, f'{score} / {max_score}', va='center', fontsize=9, color='black')

    ax.legend()
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return chart_base64

async def upload_to_firebase_storage(content: bytes, filename: str, content_type: str) -> str:
    if not bucket:
        raise HTTPException(status_code=500, detail="File storage service unavailable")
    
    try:
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        blob_path = f"resumes/{safe_filename}"
        
        blob = bucket.blob(blob_path)
        blob.metadata = {
            'contentType': content_type,
            'uploadedAt': datetime.now().isoformat(),
            'originalFilename': filename
        }
        
        blob.upload_from_string(content, content_type=content_type)
        blob.make_public()
        
        public_url = blob.public_url
        logger.info(f"✅ File uploaded successfully: {public_url}")
        
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Firebase Storage upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

# Basic health check routes
@app.get("/")
async def root():
    return {
        "message": "AIU Resume Analyzer is running!",
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "firebase": firebase_initialized,
        "spacy": nlp is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "firebase": firebase_initialized,
            "spacy": nlp is not None,
            "templates": templates is not None
        }
    }

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

# Page routes (only if templates are available)
if templates:
    @app.get("/login", response_class=HTMLResponse)
    @app.get("/", response_class=HTMLResponse)
    async def index_page(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/resume-upload", response_class=HTMLResponse)
    async def resume_upload_page(request: Request):
        return templates.TemplateResponse("resume-upload.html", {"request": request})

    @app.get("/register", response_class=HTMLResponse)
    async def register_page(request: Request):
        return templates.TemplateResponse("register.html", {"request": request})

    @app.get("/admin-login", response_class=HTMLResponse)
    async def admin_login_page(request: Request):
        return templates.TemplateResponse("admin-login.html", {"request": request})

    @app.get("/admin-dashboard", response_class=HTMLResponse)
    async def admin_dashboard(request: Request):
        return templates.TemplateResponse("admin-dashboard.html", {"request": request})

    @app.get("/forgot-password", response_class=HTMLResponse)
    async def forgot_password_page(request: Request):
        return templates.TemplateResponse("forgot-password.html", {"request": request})

# Authentication routes
@app.post("/api/register")
async def register_user(user_data: UserRegistration):
    logger.info(f"👤 User registration attempt: {user_data.email}")
    
    if user_data.password != user_data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    user_doc = {
        "full_name": user_data.full_name,
        "email": user_data.email,
        "phone": user_data.phone,
        "password_hash": hashed_password,
        "created_at": datetime.now(),
        "role": "user",
        "last_login": None,
        "address": None,
        "linkedin": None,
        "github": None,
        "bio": None
    }
    
    if db:
        try:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', user_data.email).limit(1).stream()
            
            if any(query):
                raise HTTPException(status_code=400, detail="Email already registered")
            
            doc_ref = users_ref.add(user_doc)
            user_id = doc_ref[1].id
            logger.info(f"✅ User registered in Firestore: {user_data.email}")
        except Exception as e:
            logger.error(f"❌ Error creating user: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")
    else:
        if any(u['email'] == user_data.email for u in IN_MEMORY_USERS.values()):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user_id = f"dev_user_{len(IN_MEMORY_USERS) + 1}"
        IN_MEMORY_USERS[user_id] = user_doc
        logger.info(f"✅ User registered in memory: {user_data.email}")
    
    user_info = {"user_id": user_id, "email": user_data.email, "role": "user"}
    token = create_jwt_token(user_info)
    
    return {"message": "User registered successfully", "token": token, "user": user_info}

@app.post("/api/login")
async def user_login(user_data: UserLogin):
    logger.info(f"🔐 User login attempt: {user_data.email}")
    
    try:
        if db:
            users_ref = db.collection('users')
            docs = users_ref.where('email', '==', user_data.email).limit(1).get()
            
            if len(docs) == 0:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            user_doc = docs[0].to_dict()
            user_id = docs[0].id
        else:
            user_found = None
            for uid, user in IN_MEMORY_USERS.items():
                if user['email'] == user_data.email:
                    user_found = user
                    user_id = uid
                    break
            
            if user_found is None:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            user_doc = user_found
        
        if not bcrypt.checkpw(user_data.password.encode('utf-8'), user_doc['password_hash'].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        if db:
            users_ref.document(user_id).update({"last_login": datetime.now()})
        else:
            IN_MEMORY_USERS[user_id]["last_login"] = datetime.now()
        
        user_info = {"user_id": user_id, "email": user_doc['email'], "role": user_doc.get('role', 'user')}
        token = create_jwt_token(user_info)
        
        logger.info(f"✅ User login successful: {user_data.email}")
        return {"message": "Login successful", "token": token, "user": user_info}
    
    except HTTPException:
        logger.warning(f"❌ User login failed: {user_data.email}")
        raise
    except Exception as e:
        logger.error(f"❌ Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/api/admin/login")
async def admin_login(admin_data: AdminLogin):
    logger.info(f"👑 Admin login attempt: {admin_data.email}")
    
    try:
        if db:
            admins_ref = db.collection('admins')
            docs = admins_ref.where('email', '==', admin_data.email).limit(1).get()
            
            if len(docs) == 0:
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            
            doc = docs[0]
            admin_doc = doc.to_dict()
            admin_doc['id'] = doc.id
            
            if not bcrypt.checkpw(admin_data.password.encode('utf-8'), admin_doc['password_hash'].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            
            admins_ref.document(admin_doc['id']).update({"last_login": datetime.now()})
            
            admin_info = {"user_id": admin_doc['id'], "email": admin_data.email, "role": "admin"}
            token = create_jwt_token(admin_info)
            
            logger.info(f"✅ Admin login successful: {admin_data.email}")
            return {"message": "Admin login successful", "token": token, "user": admin_info}
        else:
            if not IN_MEMORY_ADMIN:
                raise HTTPException(status_code=500, detail="Admin system not initialized")
            
            if admin_data.email != IN_MEMORY_ADMIN['email']:
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            
            if not bcrypt.checkpw(admin_data.password.encode('utf-8'), IN_MEMORY_ADMIN['password_hash'].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid admin credentials")
            
            admin_info = {"user_id": "dev_admin", "email": admin_data.email, "role": "admin"}
            token = create_jwt_token(admin_info)
            
            logger.info(f"✅ Admin login successful (dev mode): {admin_data.email}")
            return {"message": "Admin login successful (development mode)", "token": token, "user": admin_info}
    
    except HTTPException:
        logger.warning(f"❌ Admin login failed: {admin_data.email}")
        raise
    except Exception as e:
        logger.error(f"❌ Admin login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# Resume upload and analysis
@app.post("/api/resume/upload")
async def upload_resume_only(file: UploadFile = File(...), role: str = Form(...), current_user: dict = Depends(get_current_user)):
    logger.info(f"📄 Resume upload: {file.filename} by {current_user['email']}")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        pdf_content = await file.read()
        pdf_buffer = io.BytesIO(pdf_content)
        
        try:
            resume_text = pdf_reader(pdf_buffer)
            if not resume_text.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")
        
        basic_info = extract_basic_info_from_text(resume_text)
        
        timestamp = int(time.time())
        filename = f"resumes/{current_user['user_id']}/{timestamp}_{file.filename}"
        
        file_url = await upload_to_firebase_storage(
            content=pdf_content,
            filename=filename,
            content_type=file.content_type or "application/pdf"
        )
        
        # Save resume data
        resume_data = {
            "user_id": current_user['user_id'],
            "name": basic_info.get('name', 'N/A'),
            "email": basic_info.get('email', 'N/A'),
            "phone": basic_info.get('phone', 'N/A'),
            "file_name": file.filename,
            "file_url": file_url,
            "applied_role": role,
            "status": "uploaded",
            "upload_date": datetime.now().strftime("%Y-%m-%d"),
            "ats_score": None,
            "analysis_date": None,
            "analysis_data": None
        }
        
        if db:
            doc_ref = db.collection('resumes').add(resume_data)
            doc_id = doc_ref[1].id
        else:
            doc_id = f"resume_{len(IN_MEMORY_RESUMES) + 1}_{int(time.time())}"
            resume_data['id'] = doc_id
            IN_MEMORY_RESUMES[doc_id] = resume_data
        
        logger.info(f"✅ Resume uploaded successfully: {doc_id}")
        return {
            "message": "Resume uploaded successfully",
            "doc_id": doc_id,
            "basic_info": basic_info,
            "file_url": file_url,
            "applied_role": role,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "uploaded"
        }
        
    except Exception as e:
        logger.error(f"❌ Resume upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {str(e)}")

# User profile management
@app.get("/api/user/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    user_id = current_user['user_id']
    
    if db:
        try:
            user_ref = db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                raise HTTPException(status_code=404, detail="User not found")
            
            user_data = user_doc.to_dict()
            
            return {
                "user_id": user_id,
                "email": user_data.get('email', ''),
                "full_name": user_data.get('full_name', ''),
                "phone": user_data.get('phone', ''),
                "address": user_data.get('address', ''),
                "linkedin": user_data.get('linkedin', ''),
                "github": user_data.get('github', ''),
                "bio": user_data.get('bio', ''),
                "created_at": user_data.get('created_at', datetime.now()).isoformat(),
                "last_login": user_data.get('last_login', datetime.now()).isoformat() if user_data.get('last_login') else None,
                "role": user_data.get('role', 'user')
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching profile: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")
    else:
        if user_id not in IN_MEMORY_USERS:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = IN_MEMORY_USERS[user_id]
        
        return {
            "user_id": user_id,
            "email": user_data.get('email', ''),
            "full_name": user_data.get('full_name', ''),
            "phone": user_data.get('phone', ''),
            "address": user_data.get('address', ''),
            "linkedin": user_data.get('linkedin', ''),
            "github": user_data.get('github', ''),
            "bio": user_data.get('bio', ''),
            "created_at": user_data.get('created_at', datetime.now()).isoformat(),
            "last_login": user_data.get('last_login', datetime.now()).isoformat() if user_data.get('last_login') else None,
            "role": user_data.get('role', 'user')
        }

@app.get("/api/user/resumes")
async def get_user_resumes(current_user: dict = Depends(get_current_user)):
    if db:
        try:
            resumes_ref = db.collection('resumes')
            docs = resumes_ref.where('user_id', '==', current_user['user_id']).stream()
            
            results = []
            for doc in docs:
                resume_data = doc.to_dict()
                resume_data['id'] = doc.id
                results.append(resume_data)
            
            return results
        except Exception as e:
            logger.error(f"❌ Error fetching user resumes: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching user resumes: {str(e)}")
    else:
        results = []
        for resume_data in IN_MEMORY_RESUMES.values():
            if resume_data.get('user_id') == current_user['user_id']:
                results.append(resume_data)
        return results

# Admin routes
@app.get("/api/admin/stats", response_model=DashboardStats)
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        if db:
            users_ref = db.collection('users')
            user_docs = list(users_ref.stream())
            total_users = len(user_docs)
            
            resumes_ref = db.collection('resumes')
            resume_docs = list(resumes_ref.stream())
            total_resumes = len(resume_docs)
            
            analyzed_resumes = [doc for doc in resume_docs if doc.to_dict().get('ats_score') is not None]
            if analyzed_resumes:
                total_score = sum(doc.to_dict().get('ats_score', 0) for doc in analyzed_resumes)
                average_score = round(total_score / len(analyzed_resumes), 1)
            else:
                average_score = 0.0
                
        else:
            total_users = len(IN_MEMORY_USERS)
            total_resumes = len(IN_MEMORY_RESUMES)
            
            analyzed_resumes = [r for r in IN_MEMORY_RESUMES.values() if r.get('ats_score') is not None]
            if analyzed_resumes:
                total_score = sum(resume.get('ats_score', 0) for resume in analyzed_resumes)
                average_score = round(total_score / len(analyzed_resumes), 1)
            else:
                average_score = 0.0
        
        return DashboardStats(total_users=total_users, total_resumes=total_resumes, average_score=average_score)
        
    except Exception as e:
        logger.error(f"❌ Error fetching dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard stats: {str(e)}")

@app.get("/api/admin/resumes")
async def get_all_resumes(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if db:
        try:
            resumes_ref = db.collection('resumes')
            docs = resumes_ref.stream()
            
            results = []
            for doc in docs:
                resume_data = doc.to_dict()
                resume_data['id'] = doc.id
                results.append(resume_data)
            
            return results
        except Exception as e:
            logger.error(f"❌ Error fetching resumes: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching resumes: {str(e)}")
    else:
        return list(IN_MEMORY_RESUMES.values())

@app.post("/api/admin/resumes/{resume_id}/analyze")
async def analyze_resume_by_admin(resume_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    logger.info(f"🔍 Analyzing resume: {resume_id}")
    
    try:
        if db:
            doc_ref = db.collection('resumes').document(resume_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Resume not found")
            
            resume_data = doc.to_dict()
        else:
            if resume_id not in IN_MEMORY_RESUMES:
                raise HTTPException(status_code=404, detail="Resume not found")
            resume_data = IN_MEMORY_RESUMES[resume_id]
        
        if resume_data.get('status') == 'analyzed':
            raise HTTPException(status_code=400, detail="Resume already analyzed")
        
        file_url = resume_data.get('file_url')
        if not file_url:
            raise HTTPException(status_code=400, detail="Resume file URL not found")
        
        response = requests.get(file_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download resume file")
        
        pdf_buffer = io.BytesIO(response.content)
        resume_text = pdf_reader(pdf_buffer)
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        basic_info = extract_basic_info_from_text(resume_text)
        
        # Extract keywords for each category
        extraction_results = {}
        for category, keywords in KEYWORD_CATEGORIES.items():
            if category == "Candidate Profile":
                extraction_results[category] = extract_candidate_profile_keywords(basic_info)
            else:
                extraction_results[category] = extract_keywords(resume_text, keywords)
        
        # Detect sections presence
        sections_found = detect_sections_presence(resume_text)
        
        # Calculate ATS score
        ats_score, category_scores = calculate_ats_score(extraction_results, sections_found)
        
        chart_base64 = create_keyword_chart(extraction_results, sections_found, resume_data.get('applied_role', 'General'))
        
        # Update resume with analysis
        analysis_data = {
            "extraction_results": extraction_results,
            "chart": chart_base64
        }
        
        update_data = {
            "ats_score": ats_score,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_data": analysis_data,
            "status": "analyzed"
        }
        
        if db:
            db.collection('resumes').document(resume_id).update(update_data)
        else:
            IN_MEMORY_RESUMES[resume_id].update(update_data)
        
        logger.info(f"✅ Resume analyzed successfully: {resume_id} (Score: {ats_score})")
        return {
            "basic_info": basic_info,
            "extraction_results": extraction_results,
            "ats_score": ats_score,
            "category_scores": category_scores,
            "sections_found": sections_found,
            "chart": chart_base64,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "applied_role": resume_data.get('applied_role', 'General'),
            "status": "analyzed"
        }
        
    except Exception as e:
        logger.error(f"❌ Resume analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing resume: {str(e)}")

@app.get("/api/admin/resumes/{resume_id}")
async def get_resume_by_id(resume_id: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if db:
        try:
            doc_ref = db.collection('resumes').document(resume_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Resume not found")
            
            resume_data = doc.to_dict()
            resume_data['id'] = doc.id
            
            return resume_data
        except Exception as e:
            logger.error(f"❌ Error fetching resume: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching resume: {str(e)}")
    else:
        if resume_id not in IN_MEMORY_RESUMES:
            raise HTTPException(status_code=404, detail="Resume not found")
        return IN_MEMORY_RESUMES[resume_id]

# Password reset endpoints
@app.post("/api/forgot-password/check-email")
async def check_email_exists(email_data: CheckEmailRequest):
    logger.info(f"🔍 Checking email existence: {email_data.email}")
    
    try:
        if db:
            users_ref = db.collection('users')
            query = users_ref.where('email', '==', email_data.email).limit(1).get()
            exists = len(query) > 0
        else:
            exists = any(user['email'] == email_data.email for user in IN_MEMORY_USERS.values())
        
        return {
            "exists": exists,
            "message": "Account found" if exists else "No account found with this email address",
            "email": email_data.email
        }
                
    except Exception as e:
        logger.error(f"❌ Error checking email: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking email address")

# Main entry point for Railway
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Railway sets this)
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting AIU Resume Analyzer on port {port}")
    logger.info(f"🌐 Environment: Railway Production")
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info(f"🔥 Firebase initialized: {firebase_initialized}")
    logger.info(f"🧠 spaCy loaded: {nlp is not None}")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
