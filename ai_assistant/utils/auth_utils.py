#!/usr/bin/env python3
"""
User Authentication Utilities

Pure utility functions for user management:
- User creation and authentication
- Session management
- Password hashing and verification
"""

import hashlib
import secrets
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class User:
    """User data structure"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    created_at: float
    last_login: Optional[float] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        data = asdict(self)
        if self.metadata is None:
            data['metadata'] = {}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary"""
        return cls(**data)


@dataclass
class Session:
    """User session data structure"""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return asdict(self)


class UserManager:
    """Utility class for user management operations"""
    
    def __init__(self):
        """Initialize user manager"""
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.username_index: Dict[str, str] = {}  # username -> user_id
        self.email_index: Dict[str, str] = {}     # email -> user_id
    
    def create_user(self, username: str, email: str, password: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new user"""
        try:
            # Validate input
            if not username or not email or not password:
                return {'success': False, 'error': 'Missing required fields'}
            
            # Check if username/email already exists
            if username in self.username_index:
                return {'success': False, 'error': 'Username already exists'}
            
            if email in self.email_index:
                return {'success': False, 'error': 'Email already exists'}
            
            # Generate user ID
            user_id = generate_user_id()
            
            # Hash password
            salt = generate_salt()
            password_hash = hash_password(password, salt)
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                created_at=time.time(),
                metadata=metadata or {}
            )
            
            # Store user
            self.users[user_id] = user
            self.username_index[username] = user_id
            self.email_index[email] = user_id
            
            return {
                'success': True,
                'user_id': user_id,
                'user': user.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def authenticate_user(self, identifier: str, password: str) -> Dict[str, Any]:
        """Authenticate user by username/email and password"""
        try:
            # Find user by username or email
            user_id = self.username_index.get(identifier) or self.email_index.get(identifier)
            
            if not user_id:
                return {'success': False, 'error': 'User not found'}
            
            user = self.users.get(user_id)
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            if not user.is_active:
                return {'success': False, 'error': 'User account is disabled'}
            
            # Verify password
            if not verify_password(password, user.password_hash, user.salt):
                return {'success': False, 'error': 'Invalid password'}
            
            # Update last login
            user.last_login = time.time()
            
            return {
                'success': True,
                'user_id': user_id,
                'user': user.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_session(self, user_id: str, duration_hours: int = 24,
                      ip_address: Optional[str] = None, 
                      user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Create a new session for user"""
        try:
            if user_id not in self.users:
                return {'success': False, 'error': 'User not found'}
            
            session_id = generate_session_id()
            expires_at = time.time() + (duration_hours * 3600)
            
            session = Session(
                session_id=session_id,
                user_id=user_id,
                created_at=time.time(),
                expires_at=expires_at,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.sessions[session_id] = session
            
            return {
                'success': True,
                'session_id': session_id,
                'expires_at': expires_at,
                'session': session.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validate a session"""
        try:
            session = self.sessions.get(session_id)
            
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            if not session.is_active:
                return {'success': False, 'error': 'Session is inactive'}
            
            if session.is_expired():
                session.is_active = False
                return {'success': False, 'error': 'Session expired'}
            
            user = self.users.get(session.user_id)
            if not user or not user.is_active:
                return {'success': False, 'error': 'User not found or inactive'}
            
            return {
                'success': True,
                'session': session.to_dict(),
                'user': user.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def logout_session(self, session_id: str) -> Dict[str, Any]:
        """Logout/invalidate a session"""
        try:
            session = self.sessions.get(session_id)
            
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            session.is_active = False
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_id = self.username_index.get(username)
        return self.users.get(user_id) if user_id else None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        user_id = self.email_index.get(email)
        return self.users.get(user_id) if user_id else None
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user information"""
        try:
            user = self.users.get(user_id)
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Handle special fields
            if 'password' in updates:
                password = updates.pop('password')
                user.salt = generate_salt()
                user.password_hash = hash_password(password, user.salt)
            
            # Update other fields
            for key, value in updates.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            return {'success': True, 'user': user.to_dict()}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def deactivate_user(self, user_id: str) -> Dict[str, Any]:
        """Deactivate a user account"""
        try:
            user = self.users.get(user_id)
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            user.is_active = False
            
            # Deactivate all user sessions
            for session in self.sessions.values():
                if session.user_id == user_id:
                    session.is_active = False
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def list_users(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all users"""
        users: List[Dict[str, Any]] = []
        for user in self.users.values():
            if not active_only or user.is_active:
                user_data = user.to_dict()
                # Remove sensitive fields
                user_data.pop('password_hash', None)
                user_data.pop('salt', None)
                users.append(user_data)
        return users
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        expired_count = 0
        expired_sessions: List[str] = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
                expired_count += 1
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return expired_count


# Utility functions for password handling
def generate_salt(length: int = 32) -> str:
    """Generate a random salt"""
    return secrets.token_hex(length)


def hash_password(password: str, salt: str) -> str:
    """Hash a password with salt"""
    # Use PBKDF2 for secure password hashing
    password_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    
    # PBKDF2 with SHA-256, 100,000 iterations
    hashed = hashlib.pbkdf2_hmac('sha256', password_bytes, salt_bytes, 100000)
    return hashed.hex()


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify a password against stored hash"""
    computed_hash = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, stored_hash)


def generate_user_id() -> str:
    """Generate a unique user ID"""
    return f"user_{secrets.token_hex(16)}"


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"session_{secrets.token_hex(24)}"


# Utility functions for the user manager
def create_user_manager() -> UserManager:
    """Create a new user manager instance"""
    return UserManager()


def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength"""
    issues: List[str] = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        issues.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        issues.append("Password must contain at least one special character")
    
    return {
        'is_strong': len(issues) == 0,
        'issues': issues
    }


def hash_data(data: str) -> str:
    """Hash arbitrary data with SHA-256"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()