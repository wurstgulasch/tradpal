#!/usr/bin/env python3
"""
JWT-based Authentication Module for TradPal Indicator Web UI

Provides secure user authentication with JWT tokens, session management, and access control.
Replaces insecure default credentials with proper token-based authentication.
"""

import streamlit as st
import os
import json
import jwt
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import secrets


# User database file
USERS_FILE = Path(__file__).parent / "users.json"

# JWT Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))  # Generate secure random key
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24  # Token expires after 24 hours

# Blacklisted tokens (for logout)
BLACKLIST_FILE = Path(__file__).parent / "token_blacklist.json"


def load_users():
    """Load users from JSON file."""
    if not USERS_FILE.exists():
        # Create empty user database - no default admin credentials!
        users = {}
        save_users(users)
        return users

    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return {}


def save_users(users):
    """Save users to JSON file."""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        st.error(f"Error saving users: {e}")


def load_blacklist():
    """Load token blacklist."""
    if not BLACKLIST_FILE.exists():
        return set()

    try:
        with open(BLACKLIST_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('blacklisted_tokens', []))
    except Exception:
        return set()


def save_blacklist(blacklist):
    """Save token blacklist."""
    try:
        with open(BLACKLIST_FILE, 'w') as f:
            json.dump({'blacklisted_tokens': list(blacklist)}, f, indent=2)
    except Exception:
        pass


def generate_token(username: str, role: str = "user") -> str:
    """Generate JWT token for authenticated user."""
    payload = {
        'username': username,
        'role': role,
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.now(timezone.utc),
        'iss': 'tradpal_indicator'
    }

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token."""
    try:
        # Check if token is blacklisted
        blacklist = load_blacklist()
        if token in blacklist:
            return None

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Check expiration
        exp = datetime.fromtimestamp(payload['exp'], tz=timezone.utc)
        if datetime.now(timezone.utc) > exp:
            return None

        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def authenticate_user(username: str, password: str) -> tuple[bool, str, Optional[str]]:
    """Authenticate user with username and password, return token on success."""
    users = load_users()

    if username not in users:
        return False, "User not found", None

    user = users[username]
    if check_password_hash(user['password_hash'], password):
        # Generate JWT token
        token = generate_token(username, user.get('role', 'user'))
        return True, "Authentication successful", token
    else:
        return False, "Invalid password", None


def register_user(username: str, password: str, role: str = "user") -> tuple[bool, str]:
    """Register a new user."""
    users = load_users()

    if username in users:
        return False, "Username already exists"

    if len(password) < 8:  # Increased minimum password length
        return False, "Password must be at least 8 characters long"

    # Check for password complexity
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)

    if not (has_upper and has_lower and has_digit):
        return False, "Password must contain uppercase, lowercase, and numeric characters"

    users[username] = {
        "username": username,
        "password_hash": generate_password_hash(password),
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_login": None
    }

    save_users(users)
    return True, "User registered successfully"


def check_authentication() -> tuple[bool, Optional[str]]:
    """Check if user is authenticated via JWT token."""
    token = st.session_state.get('jwt_token')

    if not token:
        return False, None

    payload = verify_token(token)
    if payload:
        return True, payload.get('username')

    # Token is invalid/expired, clear session
    if 'jwt_token' in st.session_state:
        del st.session_state.jwt_token
    if 'authenticated' in st.session_state:
        del st.session_state.authenticated
    if 'username' in st.session_state:
        del st.session_state.username

    return False, None


def logout_user():
    """Logout user by blacklisting their token."""
    token = st.session_state.get('jwt_token')
    if token:
        blacklist = load_blacklist()
        blacklist.add(token)
        save_blacklist(blacklist)

    # Clear session state
    for key in ['jwt_token', 'authenticated', 'username']:
        if key in st.session_state:
            del st.session_state[key]


def login_page():
    """Display secure JWT-based login page."""
    st.title("🔐 TradPal Indicator - Secure Login")

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("Welcome Back!")

        # Create tabs for login and registration
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            st.markdown("### Sign In")

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")

                submitted = st.form_submit_button("🔑 Login", use_container_width=True)

                if submitted:
                    if username and password:
                        success, message, token = authenticate_user(username, password)
                        if success and token:
                            st.session_state.jwt_token = token
                            st.session_state.authenticated = True
                            st.session_state.username = username

                            # Update last login
                            users = load_users()
                            if username in users:
                                users[username]['last_login'] = datetime.now(timezone.utc).isoformat()
                                save_users(users)

                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter both username and password")

        with tab2:
            st.markdown("### Create Account")

            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", placeholder="Choose a strong password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

                # Password requirements info
                st.info("💡 Password must be at least 8 characters with uppercase, lowercase, and numbers")

                register_submitted = st.form_submit_button("✨ Register", use_container_width=True)

                if register_submitted:
                    if new_username and new_password and confirm_password:
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        else:
                            success, message = register_user(new_username, new_password)
                            if success:
                                st.success(message)
                                st.info("You can now login with your credentials")
                            else:
                                st.error(message)
                    else:
                        st.warning("Please fill in all fields")

    # Security notice
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>🔒 Secure JWT Authentication | TradPal Indicator v2.5</p>
            <p style='font-size: 0.8em;'>⚠️ Trading involves financial risk. This tool is for educational purposes only.</p>
        </div>
    """, unsafe_allow_html=True)


def get_user_info(username: str) -> Optional[Dict[str, Any]]:
    """Get user information."""
    users = load_users()
    return users.get(username)


def update_user_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """Update user password."""
    users = load_users()

    if username not in users:
        return False, "User not found"

    # Verify old password
    user = users[username]
    if not check_password_hash(user['password_hash'], old_password):
        return False, "Invalid current password"

    # Validate new password
    if len(new_password) < 8:
        return False, "New password must be at least 8 characters long"

    has_upper = any(c.isupper() for c in new_password)
    has_lower = any(c.islower() for c in new_password)
    has_digit = any(c.isdigit() for c in new_password)

    if not (has_upper and has_lower and has_digit):
        return False, "New password must contain uppercase, lowercase, and numeric characters"

    # Update password
    users[username]['password_hash'] = generate_password_hash(new_password)
    users[username]['updated_at'] = datetime.now(timezone.utc).isoformat()

    save_users(users)
    return True, "Password updated successfully"


def delete_user(username: str, admin_username: str) -> tuple[bool, str]:
    """Delete a user (admin only)."""
    users = load_users()

    if admin_username not in users or users[admin_username]['role'] != 'admin':
        return False, "Admin privileges required"

    if username == admin_username:
        return False, "Cannot delete your own account"

    if username not in users:
        return False, "User not found"

    del users[username]
    save_users(users)
    return True, f"User {username} deleted successfully"


def list_users(admin_username: str) -> tuple[Optional[list], str]:
    """List all users (admin only)."""
    users = load_users()

    if admin_username not in users or users[admin_username]['role'] != 'admin':
        return None, "Admin privileges required"

    user_list = []
    for username, info in users.items():
        user_list.append({
            'username': username,
            'role': info['role'],
            'created_at': info['created_at'],
            'last_login': info.get('last_login')
        })

    return user_list, "Success"


def create_initial_admin():
    """Create initial admin user if no users exist."""
    users = load_users()
    if not users:
        # Create first admin user
        admin_password = os.getenv('INITIAL_ADMIN_PASSWORD')
        if admin_password:
            success, message = register_user("admin", admin_password, "admin")
            if success:
                st.info("Initial admin user created. Please login with username 'admin'.")
            else:
                st.error(f"Failed to create admin user: {message}")
        else:
            st.warning("No users exist. Set INITIAL_ADMIN_PASSWORD environment variable to create initial admin user.")
