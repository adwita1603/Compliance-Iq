# ============================================================
# FILE: backend/app/main.py
# ============================================================
#
# WHAT IS THIS FILE?
# ------------------
# This is the ENTRY POINT of your entire backend.
# When you start the server, this file runs first.
#
# Think of this file as the "receptionist" of your application.
# Every request from the frontend comes here first.
# This file decides what to do with each request.
#
# This file handles:
#   1. Starting the server and creating the database tables
#   2. Creating default users so you can log in immediately
#   3. The login system (checking password, issuing a token)
#   4. All the API routes (what happens when the frontend calls each URL)
#
# HOW TO START THE SERVER:
# ------------------------
# Open your terminal, go to the backend/ folder, and type:
#     uvicorn app.main:app --reload
#
# Then open this URL in your browser:
#     http://localhost:8000/docs
#
# You will see a beautiful interactive page where you can test
# every single API endpoint by clicking buttons.
# The --reload flag means the server automatically restarts
# when you save any file. You don't need to restart manually.
#
# DEFAULT LOGIN CREDENTIALS (created automatically on first run):
#     admin@complianceiq.com   / Admin@1234   (role: admin)
#     officer@complianceiq.com / Officer@1234 (role: officer)
#     analyst@complianceiq.com / Analyst@1234 (role: analyst)
# ============================================================

import os
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
import logging
from app.ml_client import call_ml_pipeline
from ml.pipeline import analyze_transaction
from app.database import (
    Alert, AuditLog, Base, Transaction, User, engine, get_db
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def safe_json(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj
load_dotenv()

# ── SECURITY SETUP ──────────────────────────────────────────────────────────
# SECRET_KEY: a long random string used to sign JWT tokens.
# Anyone with this string can create fake tokens, so NEVER share it.
# We load it from the .env file.
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE-THIS-IN-PRODUCTION")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8

# pwd_context: handles bcrypt password hashing.
# bcrypt is a one-way scrambler. Given "password123", it produces
# "$2b$12$abc...xyz" — a string that looks like garbage.
# You can check if "password123" matches the garbage, but you
# cannot reverse the garbage to find "password123".
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# oauth2_scheme: tells FastAPI where to look for the JWT token in requests.
# The frontend sends it as a header: "Authorization: Bearer <token>"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")



# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """Convert a plain password into a bcrypt hash for safe storage."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check if a plain password matches a stored hash.
    Returns True if they match, False otherwise.
    Called during login to verify the entered password.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, email: str, role: str) -> str:
    """
    Create a JWT token — a signed "ID badge" valid for 8 hours.

    The token contains: user_id, email, role, expiry time.
    It is signed with SECRET_KEY so it cannot be tampered with.
    Anyone with this token can act as this user until it expires.

    Think of it like a concert wristband:
    - It proves you paid (logged in legitimately)
    - It expires at midnight (8 hours)
    - If someone forges it, security (our server) catches it
    """
    payload = {
        "sub": user_id,       # "sub" = subject (who this token is for)
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI "dependency" — reads the JWT token from the request
    and returns the logged-in user.

    Add this to any route that requires login:
        current_user: User = Depends(get_current_user)

    FastAPI calls this automatically before running your route.
    If the token is missing or invalid, it blocks the request with a 401 error.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Token is invalid or has expired. Please log in again.",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated.")
    return user


def require_roles(allowed_roles: list):
    """
    FastAPI "dependency factory" — blocks a route if the user's
    role is not in the allowed list.

    HOW TO USE:
        @app.put("/alerts/{id}/resolve")
        def resolve(
            id: str,
            current_user: User = Depends(require_roles(["officer","admin"]))
        ):
            ...

    If an analyst tries to hit this route, they get:
        403 Forbidden: "Access denied. Required roles: [officer, admin]"

    IMPORTANT: This check happens on the SERVER — not just in the UI.
    Hiding buttons in the frontend is cosmetic only. The real
    enforcement is here, on every API call.
    """
    def checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required: {allowed_roles}. Your role: {current_user.role}",
            )
        return current_user
    return checker


def write_audit_log(
    db: Session,
    action: str,
    detail: str,
    user_id: str = None,
    ip: str = None,
):
    """
    Write one row to the audit_logs table.
    Call this in any route where you want to record what happened.

    Example:
        write_audit_log(db, "alert_resolved", "Alert abc123 resolved", user_id=current_user.id)
    """
    log = AuditLog(
        user_id=user_id,
        action=action,
        detail=detail,
        ip_address=ip,
    )
    db.add(log)
    db.commit()


# ── REQUEST / RESPONSE SCHEMAS ───────────────────────────────────────────────
# Pydantic models define the shape of JSON data your API accepts and returns.
# FastAPI validates all incoming data against these automatically.
# If the data doesn't match, FastAPI returns a 422 error automatically.

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str

class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "analyst"   # default to lowest privilege

class AlertResolveRequest(BaseModel):
    notes: str = ""   # officer can add optional notes

class TransactionCheckRequest(BaseModel):
    """What you send when submitting a transaction for checking."""
    sender_account: str
    receiver_account: str
    amount: float
    transaction_type: str = "transfer"
    hour_of_day: int = 12       # default noon if not provided
    tx_count_7d: int = 1        # default 1 transaction this week
    kyc_verified: bool = True

class ChatbotRequest(BaseModel):
    question: str


# ── APP STARTUP ──────────────────────────────────────────────────────────────
# Code inside @asynccontextmanager runs ONCE when the server starts.
# Use it for one-time setup: creating tables, creating the default admin.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ComplianceIQ starting up...")

    # Create all database tables if they don't exist yet.
    # This is safe to run every time — if tables exist, it does nothing.
    Base.metadata.create_all(bind=engine)
    logger.info("  Database tables ready.")

    # Create default users if none exist yet.
    # This means on your very first run, you can immediately log in
    # without manually creating users in the database.
    db = next(get_db())
    try:
        if not db.query(User).filter(User.email == "admin@complianceiq.com").first():
            db.add(User(
                email="admin@complianceiq.com",
                hashed_password=hash_password("Admin@1234"),
                role="admin",
            ))
            db.add(User(
                email="officer@complianceiq.com",
                hashed_password=hash_password("Officer@1234"),
                role="officer",
            ))
            db.add(User(
                email="analyst@complianceiq.com",
                hashed_password=hash_password("Analyst@1234"),
                role="analyst",
            ))
            db.commit()
            logger.info("  Default users created:")
            logger.info("    admin@complianceiq.com   / Admin@1234")
            logger.info("    officer@complianceiq.com / Officer@1234")
            logger.info("    analyst@complianceiq.com / Analyst@1234")
        else:
            logger.info("  Users already exist — skipping.")
    finally:
        db.close()

    logger.info("  Ready. Visit: http://localhost:8000/docs")
    yield  # Server runs here. Everything above = startup. Below = shutdown.
    logger.info("ComplianceIQ shutting down.")


# ── CREATE THE APP ────────────────────────────────────────────────────────────
app = FastAPI(
    title="ComplianceIQ API",
    description="AI-powered compliance monitoring for Indian fintech.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS CONFIGURATION ────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing.
# Without this, your React frontend (on port 5173) CANNOT call this server
# (on port 8000). Browsers block it as a security measure.
# This tells the browser: "yes, requests from these origins are allowed."
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",            # React dev server (Vite)
        "http://localhost:3000",            # React dev server (CRA)
        "https://complianceiq.vercel.app",  # Deployed frontend (update later)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════
# API ROUTES
# Each function below handles one type of request from the frontend.
# The decorator (@app.get, @app.post, etc.) says:
#   - Which HTTP method (GET = read data, POST = send data, PUT = update)
#   - Which URL path ("/auth/login", "/alerts", etc.)
# ════════════════════════════════════════════════════════════════════════════


# ── HEALTH CHECK ─────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    """
    Health check endpoint.
    Railway and Vercel use this to check if your service is alive.
    Returns a simple JSON response confirming the server is running.
    """
    return {
        "status": "running",
        "app": "ComplianceIQ",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.get("/ml/health", tags=["health"])
def ml_health():
    """
    Checks whether the ML pipeline is available and the model can be loaded.
    """
    try:
        sample_transaction = {
            "sender_account": "TEST001",
            "receiver_account": "TEST002",
            "amount": 1000,
            "hour_of_day": 12,
            "tx_count_7d": 1,
            "kyc_verified": True,
        }

        return {
        "status": "healthy",
        "pipeline": "ML Service Connected"
    }

        

    except Exception as e:
        return {
            "status": "unhealthy",
            "pipeline": "failed",
            "error": str(e),
        }


# ── LOGIN ─────────────────────────────────────────────────────────────────────
@app.post("/auth/login", response_model=LoginResponse, tags=["auth"])
#def login(request: LoginRequest, req: Request, db: Session = Depends(get_db)):
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    req: Request = None,
    db: Session = Depends(get_db)
):
    """
    Login endpoint.

    What happens step by step:
    1. Find the user by their email in the database
    2. Check if their account is locked (too many wrong passwords)
    3. Verify their password against the stored bcrypt hash
    4. If correct: issue a JWT token valid for 8 hours
    5. Write a "login_success" event to the audit log

    Try it in /docs:
        POST /auth/login
        Body: {"email": "admin@complianceiq.com", "password": "Admin@1234"}
        Response: {"access_token": "eyJ...", "role": "admin", ...}
    """
    ip = req.client.host

    # Step 1: Find user by email
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        # We give the SAME error whether email OR password is wrong.
        # This prevents attackers from figuring out which emails exist.
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Step 2: Check if account is active
    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="This account has been deactivated. Contact your admin.",
        )

    # Step 3: Check if account is temporarily locked
    if user.locked_until and user.locked_until > datetime.utcnow():
        mins = int((user.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Account locked. Try again in {mins} minute(s).",
        )

    # Step 4: Verify password
    if not verify_password(form_data.password, user.hashed_password):
        user.failed_login_count += 1
        if user.failed_login_count >= 5:
            # Lock the account for 15 minutes
            user.locked_until = datetime.utcnow() + timedelta(minutes=15)
            user.failed_login_count = 0
            db.commit()
            raise HTTPException(
                status_code=429,
                detail="Account locked for 15 minutes after 5 failed attempts.",
            )
        db.commit()
        write_audit_log(db, "login_failed", f"Failed attempt {user.failed_login_count}/5", user_id=user.id, ip=ip)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Step 5: Login successful!
    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = datetime.utcnow()
    db.commit()

    token = create_access_token(user.id, user.email, user.role)
    write_audit_log(db, "login_success", f"Successful login from {ip}", user_id=user.id, ip=ip)

    return LoginResponse(
        access_token=token,
        user_id=user.id,
        email=user.email,
        role=user.role,
    )


# ── GET MY PROFILE ────────────────────────────────────────────────────────────
@app.get("/auth/me", tags=["auth"])
def get_my_profile(current_user: User = Depends(get_current_user)):
    """
    Returns the currently logged-in user's profile.
    The frontend calls this on page load to know who is logged in.

    Requires: a valid JWT token in the Authorization header.
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "last_login_at": current_user.last_login_at,
    }


# ── CHECK A TRANSACTION ───────────────────────────────────────────────────────
@app.post("/transactions/check", tags=["transactions"])
async def check_transaction(
    request: TransactionCheckRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Submit a transaction for compliance checking using:
    - RBI/PMLA rule engine
    - XGBoost fraud detection
    - SHAP explainability
    """

    # ── SAVE TRANSACTION TO DATABASE ─────────────────────────────
    tx = Transaction(
        sender_account=request.sender_account,
        receiver_account=request.receiver_account,
        amount=request.amount,
        transaction_type=request.transaction_type,
        hour_of_day=request.hour_of_day,
        tx_count_7d=request.tx_count_7d,
        kyc_verified=request.kyc_verified,
        status="pending",
    )

    db.add(tx)
    db.commit()
    db.refresh(tx)

    # ── PREPARE DATA FOR ML PIPELINE ─────────────────────────────
    tx_data = {
        "amount": request.amount,
        "hour_of_day": request.hour_of_day,
        "tx_count_7d": request.tx_count_7d,
        "kyc_verified": request.kyc_verified,
    }

    # ── RUN ML + RULE ENGINE PIPELINE ────────────────────────────
    result = analyze_transaction(tx_data)

        # ── EXTRACT RESULTS ──────────────────────────────────────────
    is_flagged = result.get("flagged", False)
    risk_level = result.get("risk_level", "low")
    risk_score = result.get("ml_probability", 0.0)

    violations = result.get("rule_violations", [])
    shap_explanation = result.get("shap_explanation", {})
    ai_alert = result.get("summary", "No alert generated.")
    # ── UPDATE TRANSACTION ───────────────────────────────────────
    tx.status = "flagged" if is_flagged else "clean"
    tx.risk_score = risk_score
    tx.checked_at = datetime.utcnow()

    # ── CREATE ALERT IF FLAGGED ──────────────────────────────────
    alert = None

    if is_flagged:

        violation_type = (
            violations[0]["rule_name"]
            if violations else
            "ML Fraud Detection"
        )

        regulation_cited = (
            violations[0].get("regulation_source")
            if violations else
            "AI Risk Engine"
        )

        alert = Alert(
            transaction_id=tx.id,
            risk_level=result["risk_level"].lower(),
            violation_type=violation_type,
            regulation_cited=regulation_cited,
            ai_explanation=ai_alert,
            shap_features_json=json.dumps(safe_json(shap_explanation)),
            status="open",
        )

        db.add(alert)

    # ── SAVE CHANGES ─────────────────────────────────────────────
    db.commit()

    if alert:
        db.refresh(alert)
    logger.info(
    "Transaction %s completed | Flagged=%s",
    tx.id,
    is_flagged,
    )


    # ── RETURN RESPONSE ──────────────────────────────────────────
    return {
        "transaction_id": tx.id,
        "flagged": is_flagged,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "violations": violations,
        "shap_explanation": shap_explanation,
        "alert_id": alert.id if alert else None,
        "message": (
            "Transaction flagged and alert created."
            if is_flagged else
            "Transaction appears clean."
        ),
    }

# ── GET ALL ALERTS ─────────────────────────────────────────────────────────
@app.get("/alerts", tags=["alerts"])
def list_alerts(
    status: str = None,
    risk_level: str = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get all alerts. Any logged-in user can call this.

    Optional filters (add as URL query params):
        ?status=open           → only open alerts
        ?risk_level=high       → only high-risk alerts
        ?status=open&risk_level=high  → both filters

    Pagination:
        ?limit=20&offset=0     → first 20 alerts
        ?limit=20&offset=20    → next 20 alerts

    Try it in /docs:
        GET /alerts
        GET /alerts?status=open
        GET /alerts?risk_level=high
    """
    query = db.query(Alert)
    if status:
        query = query.filter(Alert.status == status)
    if risk_level:
        query = query.filter(Alert.risk_level == risk_level)

    total = query.count()
    alerts = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "alerts": [
            {
                "id": a.id,
                "transaction_id": a.transaction_id,
                "risk_level": a.risk_level,
                "violation_type": a.violation_type,
                "regulation_cited": a.regulation_cited,
                "status": a.status,
                "created_at": a.created_at.isoformat(),
                "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
            }
            for a in alerts
        ],
    }


# ── GET ONE ALERT (full details) ─────────────────────────────────────────────
@app.get("/alerts/{alert_id}", tags=["alerts"])
def get_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get full details of one alert including the AI explanation,
    SHAP features, and the transaction that triggered it.

    This is what the Alert Detail page calls.

    Try it in /docs:
        GET /alerts/{alert_id}
    Replace {alert_id} with a real ID from GET /alerts.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")

    tx = db.query(Transaction).filter(Transaction.id == alert.transaction_id).first()

    return {
        "id": alert.id,
        "risk_level": alert.risk_level,
        "violation_type": alert.violation_type,
        "regulation_cited": alert.regulation_cited,
        "ai_explanation": alert.ai_explanation,
        "shap_features": alert.shap_features_json,
        "status": alert.status,
        "resolution_notes": alert.resolution_notes,
        "created_at": alert.created_at.isoformat(),
        "transaction": {
            "id": tx.id,
            "sender_account": tx.sender_account,
            "receiver_account": tx.receiver_account,
            "amount": tx.amount,
            "hour_of_day": tx.hour_of_day,
            "tx_count_7d": tx.tx_count_7d,
            "kyc_verified": tx.kyc_verified,
            "risk_score": tx.risk_score,
        } if tx else None,
    }


# ── RESOLVE AN ALERT ───────────────────────────────────────────────────────
@app.put("/alerts/{alert_id}/resolve", tags=["alerts"])
def resolve_alert(
    alert_id: str,
    body: AlertResolveRequest,
    # require_roles means ONLY officer or admin can call this route
    current_user: User = Depends(require_roles(["officer", "admin"])),
    db: Session = Depends(get_db),
):
    """
    Mark an alert as resolved.
    ONLY officers and admins can do this (analysts get 403 Forbidden).
    Also writes to the audit log.

    Try it in /docs:
        First log in as officer@complianceiq.com / Officer@1234
        Then: PUT /alerts/{alert_id}/resolve
        Body: {"notes": "Verified with customer. Legitimate payment."}
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    if alert.status == "resolved":
        raise HTTPException(status_code=400, detail="Alert is already resolved.")

    alert.status = "resolved"
    alert.resolved_by = current_user.id
    alert.resolved_at = datetime.utcnow()
    alert.resolution_notes = body.notes
    db.commit()

    write_audit_log(
        db,
        action="alert_resolved",
        detail=f"Alert {alert_id} resolved by {current_user.email}. Notes: {body.notes}",
        user_id=current_user.id,
    )

    return {"message": "Alert resolved.", "alert_id": alert_id}


# ── ESCALATE AN ALERT ──────────────────────────────────────────────────────
@app.put("/alerts/{alert_id}/escalate", tags=["alerts"])
def escalate_alert(
    alert_id: str,
    current_user: User = Depends(require_roles(["officer", "admin"])),
    db: Session = Depends(get_db),
):
    """Mark an alert as escalated for senior review."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")

    alert.status = "escalated"
    db.commit()
    write_audit_log(db, "alert_escalated", f"Alert {alert_id} escalated", user_id=current_user.id)
    return {"message": "Alert escalated.", "alert_id": alert_id}


# ── DASHBOARD SUMMARY ────────────────────────────────────────────────────────
@app.get("/dashboard/summary", tags=["dashboard"])
def dashboard_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns the 4 numbers shown on the dashboard summary cards:
        - alerts_today     → how many alerts were created today
        - high_risk_open   → how many high-risk alerts are still open
        - resolved_week    → how many alerts were resolved this week
        - total_open       → total open alerts right now

    The frontend calls this to populate the 4 stat cards at the top.
    """
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = datetime.utcnow() - timedelta(days=7)

    return {
        "alerts_today": db.query(Alert).filter(Alert.created_at >= today).count(),
        "high_risk_open": db.query(Alert).filter(
            Alert.risk_level == "high",
            Alert.status == "open",
        ).count(),
        "resolved_week": db.query(Alert).filter(
            Alert.status == "resolved",
            Alert.resolved_at >= week_ago,
        ).count(),
        "total_open": db.query(Alert).filter(Alert.status == "open").count(),
        "system_status": "operational",
    }


# ── CREATE USER (admin only) ──────────────────────────────────────────────
@app.post("/users", tags=["admin"], status_code=201)
def create_user(
    body: CreateUserRequest,
    current_user: User = Depends(require_roles(["admin"])),
    db: Session = Depends(get_db),
):
    """
    Create a new user account. ONLY admins can do this.

    Try it in /docs:
        Log in as admin@complianceiq.com
        POST /users
        Body: {"email": "john@company.com", "password": "Pass@1234", "role": "analyst"}
    """
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already exists.")
    if body.role not in ["analyst", "officer", "admin"]:
        raise HTTPException(status_code=400, detail="Role must be: analyst, officer, or admin.")

    new_user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        role=body.role,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    write_audit_log(
        db, "user_created",
        f"User {body.email} created with role {body.role}",
        user_id=current_user.id,
    )

    return {
        "message": "User created.",
        "user_id": new_user.id,
        "email": new_user.email,
        "role": new_user.role,
    }


# ── LIST ALL USERS (admin only) ────────────────────────────────────────────
@app.get("/users", tags=["admin"])
def list_users(
    current_user: User = Depends(require_roles(["admin"])),
    db: Session = Depends(get_db),
):
    """Get all users. ONLY admins can do this."""
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        }
        for u in users
    ]


# ── DEACTIVATE USER (admin only) ────────────────────────────────────────────
@app.put("/users/{user_id}/deactivate", tags=["admin"])
def deactivate_user(
    user_id: str,
    current_user: User = Depends(require_roles(["admin"])),
    db: Session = Depends(get_db),
):
    """Deactivate a user account. ONLY admins can do this."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account.")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.is_active = False
    db.commit()
    write_audit_log(db, "user_deactivated", f"User {user.email} deactivated", user_id=current_user.id)
    return {"message": f"User {user.email} deactivated."}


# ── AUDIT LOGS (admin only) ───────────────────────────────────────────────
@app.get("/audit-logs", tags=["admin"])
def get_audit_logs(
    limit: int = 100,
    action_filter: str = None,
    current_user: User = Depends(require_roles(["admin"])),
    db: Session = Depends(get_db),
):
    """Get audit logs. ONLY admins can do this."""
    query = db.query(AuditLog)
    if action_filter:
        query = query.filter(AuditLog.action == action_filter)

    logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": l.id,
            "user_id": l.user_id,
            "action": l.action,
            "detail": l.detail,
            "ip_address": l.ip_address,
            "timestamp": l.timestamp.isoformat(),
        }
        for l in logs
    ]


# ── CHATBOT (stub — Person 2 fills this in later) ───────────────────────────
@app.post("/chatbot/ask", tags=["chatbot"])
def chatbot_ask(
    body: ChatbotRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from ml.chatbot import answer_question
    result = answer_question(body.question)
    return result


# ── START THE SERVER (when running this file directly) ─────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ComplianceIQ server...")
    logger.info("API docs: http://localhost:8000/docs")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
