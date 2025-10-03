#!/usr/bin/env python3
"""
Advanced Authentication Utilities

Enterprise-grade authentication system with:
- UUID-based user identification
- Multiple contact methods per user
- Profile management with images
- Enhanced security features
- Comprehensive audit logging
"""

import hashlib
import secrets
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class ContactType(Enum):
    """Contact method types"""
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"


class UserRole(Enum):
    """User role definitions"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SUSPENDED = "suspended"


class SessionStatus(Enum):
    """Session status types"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


@dataclass
class ContactMethod:
    """User contact method"""
    contact_id: str
    contact_type: ContactType
    value: str
    is_primary: bool = False
    is_verified: bool = False
    created_at: float = field(default_factory=time.time)
    verified_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contact_id': self.contact_id,
            'contact_type': self.contact_type.value,
            'value': self.value,
            'is_primary': self.is_primary,
            'is_verified': self.is_verified,
            'created_at': self.created_at,
            'verified_at': self.verified_at
        }


@dataclass
class UserProfile:
    """Enhanced user profile"""
    user_id: str
    username: str
    password_hash: str
    salt: str
    role: UserRole
    contact_methods: Dict[str, ContactMethod] = field(default_factory=dict)
    profile_picture_path: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    timezone: str = "UTC"
    locale: str = "en_US"
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[float] = None
    password_changed_at: float = field(default_factory=time.time)
    last_login_at: Optional[float] = None
    last_login_ip: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role.value,
            'contact_methods': {k: v.to_dict() for k, v in self.contact_methods.items()},
            'profile_picture_path': self.profile_picture_path,
            'display_name': self.display_name,
            'bio': self.bio,
            'timezone': self.timezone,
            'locale': self.locale,
            'two_factor_enabled': self.two_factor_enabled,
            'failed_login_attempts': self.failed_login_attempts,
            'account_locked_until': self.account_locked_until,
            'password_changed_at': self.password_changed_at,
            'last_login_at': self.last_login_at,
            'last_login_ip': self.last_login_ip,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_active': self.is_active,
            'metadata': self.metadata.copy()
        }
        
        if include_sensitive:
            data.update({
                'password_hash': self.password_hash,
                'salt': self.salt,
                'two_factor_secret': self.two_factor_secret
            })
        
        return data


@dataclass
class SecureSession:
    """Enhanced session with security features"""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    last_activity_at: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    security_level: int = 1  # 1=basic, 2=elevated, 3=admin
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return time.time() > self.expires_at
    
    def is_active(self) -> bool:
        """Check if session is active"""
        return self.status == SessionStatus.ACTIVE and not self.is_expired()
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'last_activity_at': self.last_activity_at,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'status': self.status.value,
            'security_level': self.security_level,
            'metadata': self.metadata.copy()
        }


@dataclass
class AuditLogEntry:
    """Security audit log entry"""
    entry_id: str
    user_id: Optional[str]
    action: str
    resource: str
    status: str  # success, failure, warning
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entry_id': self.entry_id,
            'user_id': self.user_id,
            'action': self.action,
            'resource': self.resource,
            'status': self.status,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'details': self.details.copy(),
            'timestamp': self.timestamp
        }


class AdvancedAuthManager:
    """Enterprise authentication manager"""
    
    def __init__(self):
        """Initialize authentication manager"""
        self.users: Dict[str, UserProfile] = {}
        self.sessions: Dict[str, SecureSession] = {}
        self.audit_log: List[AuditLogEntry] = []
        self.username_index: Dict[str, str] = {}  # username -> user_id
        self.email_index: Dict[str, str] = {}     # email -> user_id
        self.phone_index: Dict[str, str] = {}     # phone -> user_id
        
        # Security configuration
        self.max_login_attempts = 5
        self.lockout_duration = 3600  # 1 hour
        self.password_min_length = 12
        self.password_complexity_required = True
        self.session_timeout = 86400  # 24 hours
        self.pbkdf2_iterations = 150000
    
    def _log_audit(self, action: str, resource: str, status: str,
                   user_id: Optional[str] = None, ip_address: Optional[str] = None,
                   user_agent: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Log security audit event"""
        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            user_id=user_id,
            action=action,
            resource=resource,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        self.audit_log.append(entry)
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return str(uuid.uuid4())
    
    def _generate_contact_id(self) -> str:
        """Generate unique contact ID"""
        return str(uuid.uuid4())
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password using PBKDF2"""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            self.pbkdf2_iterations
        ).hex()
    
    def _generate_salt(self) -> str:
        """Generate cryptographic salt"""
        return secrets.token_hex(32)
    
    def _validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password meets security requirements"""
        issues: List[str] = []
        
        if len(password) < self.password_min_length:
            issues.append(f"Password must be at least {self.password_min_length} characters")
        
        if self.password_complexity_required:
            if not any(c.isupper() for c in password):
                issues.append("Password must contain uppercase letters")
            if not any(c.islower() for c in password):
                issues.append("Password must contain lowercase letters")
            if not any(c.isdigit() for c in password):
                issues.append("Password must contain digits")
            if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                issues.append("Password must contain special characters")
        
        return {
            'is_strong': len(issues) == 0,
            'issues': issues
        }
    
    def _is_account_locked(self, user: UserProfile) -> bool:
        """Check if account is locked due to failed attempts"""
        if user.account_locked_until is None:
            return False
        return time.time() < user.account_locked_until
    
    def _lock_account(self, user: UserProfile) -> None:
        """Lock account due to failed login attempts"""
        user.account_locked_until = time.time() + self.lockout_duration
        user.failed_login_attempts = 0
        
        self._log_audit(
            action="account_locked",
            resource="user_account",
            status="warning",
            user_id=user.user_id,
            details={'lockout_duration': self.lockout_duration}
        )
    
    def create_user(self, username: str, primary_email: str, password: str,
                   role: UserRole = UserRole.USER, display_name: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create new user with enhanced profile"""
        try:
            # Validate inputs
            if not username or not primary_email or not password:
                return {'success': False, 'error': 'Missing required fields'}
            
            if username in self.username_index:
                return {'success': False, 'error': 'Username already exists'}
            
            if primary_email in self.email_index:
                return {'success': False, 'error': 'Email already registered'}
            
            # Validate password strength
            password_check = self._validate_password_strength(password)
            if not password_check['is_strong']:
                return {
                    'success': False,
                    'error': 'Password does not meet requirements',
                    'password_issues': password_check['issues']
                }
            
            # Generate user ID and security data
            user_id = self._generate_user_id()
            salt = self._generate_salt()
            password_hash = self._hash_password(password, salt)
            
            # Create primary email contact
            email_contact = ContactMethod(
                contact_id=self._generate_contact_id(),
                contact_type=ContactType.EMAIL,
                value=primary_email,
                is_primary=True,
                is_verified=False
            )
            
            # Create user profile
            user = UserProfile(
                user_id=user_id,
                username=username,
                password_hash=password_hash,
                salt=salt,
                role=role,
                display_name=display_name,
                metadata=metadata or {}
            )
            
            user.contact_methods[email_contact.contact_id] = email_contact
            
            # Store user and update indices
            self.users[user_id] = user
            self.username_index[username] = user_id
            self.email_index[primary_email] = user_id
            
            self._log_audit(
                action="user_created",
                resource="user_account",
                status="success",
                user_id=user_id,
                details={'username': username, 'role': role.value}
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'user': user.to_dict()
            }
            
        except Exception as e:
            self._log_audit(
                action="user_creation_failed",
                resource="user_account",
                status="failure",
                details={'error': str(e), 'username': username}
            )
            return {'success': False, 'error': str(e)}
    
    def authenticate_user(self, identifier: str, password: str,
                         ip_address: Optional[str] = None,
                         user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Authenticate user with enhanced security"""
        try:
            # Find user by username or email
            user_id = self.username_index.get(identifier) or self.email_index.get(identifier)
            if not user_id:
                self._log_audit(
                    action="login_failed",
                    resource="authentication",
                    status="failure",
                    ip_address=ip_address,
                    details={'identifier': identifier, 'reason': 'user_not_found'}
                )
                return {'success': False, 'error': 'Invalid credentials'}
            
            user = self.users[user_id]
            
            # Check if account is locked
            if self._is_account_locked(user):
                self._log_audit(
                    action="login_blocked",
                    resource="authentication",
                    status="warning",
                    user_id=user_id,
                    ip_address=ip_address,
                    details={'reason': 'account_locked'}
                )
                return {'success': False, 'error': 'Account is locked'}
            
            # Check if account is active
            if not user.is_active:
                self._log_audit(
                    action="login_blocked",
                    resource="authentication",
                    status="warning",
                    user_id=user_id,
                    ip_address=ip_address,
                    details={'reason': 'account_inactive'}
                )
                return {'success': False, 'error': 'Account is inactive'}
            
            # Verify password
            password_hash = self._hash_password(password, user.salt)
            if not secrets.compare_digest(password_hash, user.password_hash):
                user.failed_login_attempts += 1
                
                if user.failed_login_attempts >= self.max_login_attempts:
                    self._lock_account(user)
                
                self._log_audit(
                    action="login_failed",
                    resource="authentication",
                    status="failure",
                    user_id=user_id,
                    ip_address=ip_address,
                    details={'reason': 'invalid_password', 'attempt': user.failed_login_attempts}
                )
                return {'success': False, 'error': 'Invalid credentials'}
            
            # Successful authentication
            user.failed_login_attempts = 0
            user.last_login_at = time.time()
            user.last_login_ip = ip_address
            user.updated_at = time.time()
            
            self._log_audit(
                action="login_success",
                resource="authentication",
                status="success",
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'user': user.to_dict()
            }
            
        except Exception as e:
            self._log_audit(
                action="authentication_error",
                resource="authentication",
                status="failure",
                ip_address=ip_address,
                details={'error': str(e)}
            )
            return {'success': False, 'error': str(e)}
    
    def create_session(self, user_id: str, security_level: int = 1,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      duration_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Create secure session"""
        try:
            if user_id not in self.users:
                return {'success': False, 'error': 'User not found'}
            
            user = self.users[user_id]
            if not user.is_active:
                return {'success': False, 'error': 'User account inactive'}
            
            session_id = self._generate_session_id()
            current_time = time.time()
            expires_at = current_time + (duration_seconds or self.session_timeout)
            
            session = SecureSession(
                session_id=session_id,
                user_id=user_id,
                created_at=current_time,
                expires_at=expires_at,
                last_activity_at=current_time,
                ip_address=ip_address,
                user_agent=user_agent,
                security_level=security_level
            )
            
            self.sessions[session_id] = session
            
            self._log_audit(
                action="session_created",
                resource="session",
                status="success",
                user_id=user_id,
                ip_address=ip_address,
                details={'session_id': session_id, 'security_level': security_level}
            )
            
            return {
                'success': True,
                'session_id': session_id,
                'expires_at': expires_at,
                'session': session.to_dict()
            }
            
        except Exception as e:
            self._log_audit(
                action="session_creation_failed",
                resource="session",
                status="failure",
                user_id=user_id,
                details={'error': str(e)}
            )
            return {'success': False, 'error': str(e)}
    
    def add_contact_method(self, user_id: str, contact_type: ContactType,
                          value: str, is_primary: bool = False) -> Dict[str, Any]:
        """Add contact method to user profile"""
        try:
            if user_id not in self.users:
                return {'success': False, 'error': 'User not found'}
            
            user = self.users[user_id]
            
            # Check if contact already exists
            if contact_type == ContactType.EMAIL and value in self.email_index:
                return {'success': False, 'error': 'Email already registered'}
            
            if contact_type == ContactType.PHONE and value in self.phone_index:
                return {'success': False, 'error': 'Phone already registered'}
            
            # If setting as primary, remove primary flag from others
            if is_primary:
                for contact in user.contact_methods.values():
                    if contact.contact_type == contact_type:
                        contact.is_primary = False
            
            contact = ContactMethod(
                contact_id=self._generate_contact_id(),
                contact_type=contact_type,
                value=value,
                is_primary=is_primary
            )
            
            user.contact_methods[contact.contact_id] = contact
            user.updated_at = time.time()
            
            # Update indices
            if contact_type == ContactType.EMAIL:
                self.email_index[value] = user_id
            elif contact_type == ContactType.PHONE:
                self.phone_index[value] = user_id
            
            self._log_audit(
                action="contact_added",
                resource="user_profile",
                status="success",
                user_id=user_id,
                details={'contact_type': contact_type.value, 'is_primary': is_primary}
            )
            
            return {
                'success': True,
                'contact_id': contact.contact_id,
                'contact': contact.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_user_audit_log(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries for specific user"""
        user_entries = [
            entry.to_dict() for entry in self.audit_log
            if entry.user_id == user_id
        ]
        return sorted(user_entries, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics"""
        current_time = time.time()
        
        active_sessions = sum(1 for s in self.sessions.values() if s.is_active())
        locked_accounts = sum(1 for u in self.users.values() if self._is_account_locked(u))
        recent_failures = sum(
            1 for entry in self.audit_log
            if entry.action == "login_failed" and current_time - entry.timestamp < 3600
        )
        
        return {
            'total_users': len(self.users),
            'active_users': sum(1 for u in self.users.values() if u.is_active),
            'locked_accounts': locked_accounts,
            'active_sessions': active_sessions,
            'recent_login_failures': recent_failures,
            'audit_log_entries': len(self.audit_log),
            'two_factor_enabled_users': sum(1 for u in self.users.values() if u.two_factor_enabled)
        }


# Utility functions
def create_advanced_auth_manager() -> AdvancedAuthManager:
    """Create new advanced authentication manager"""
    return AdvancedAuthManager()


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure token"""
    return secrets.token_urlsafe(length)


def validate_email_format(email: str) -> bool:
    """Basic email format validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone_format(phone: str) -> bool:
    """Basic phone format validation"""
    import re
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    # Check if it's between 10-15 digits (international format)
    return 10 <= len(digits_only) <= 15