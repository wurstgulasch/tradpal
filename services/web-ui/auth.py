#!/usr/bin/env python3
"""
Authentication Module for TradPal Indicator Web UI

Provides user authentication, session management, and access control.
"""

import streamlit as st
import os
import json
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta


# User database file
USERS_FILE = Path(__file__).parent / "users.json"

# Default admin credentials (change in production!)
DEFAULT_ADMIN = {
    "username": "admin",
    "password_hash": generate_password_hash("admin123"),
    "role": "admin",
    "created_at": datetime.now().isoformat()
}


def load_users():
    """Load users from JSON file."""
    if not USERS_FILE.exists():
        # Create default admin user
        users = {"admin": DEFAULT_ADMIN}
        save_users(users)
        return users
    
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return {"admin": DEFAULT_ADMIN}


def save_users(users):
    """Save users to JSON file."""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        st.error(f"Error saving users: {e}")


def authenticate_user(username, password):
    """Authenticate user with username and password."""
    users = load_users()
    
    if username not in users:
        return False, "User not found"
    
    user = users[username]
    if check_password_hash(user['password_hash'], password):
        return True, "Authentication successful"
    else:
        return False, "Invalid password"


def register_user(username, password, role="user"):
    """Register a new user."""
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    users[username] = {
        "username": username,
        "password_hash": generate_password_hash(password),
        "role": role,
        "created_at": datetime.now().isoformat()
    }
    
    save_users(users)
    return True, "User registered successfully"


def check_authentication():
    """Check if user is authenticated."""
    return st.session_state.get('authenticated', False)


def login_page():
    """Display login page."""
    st.title("🔐 TradPal Indicator - Login")
    
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
                        success, message = authenticate_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Please enter both username and password")
            
            # Default credentials info (remove in production)
            st.info("ℹ️ Default credentials: admin / admin123")
        
        with tab2:
            st.markdown("### Create Account")
            
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                
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
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>🔒 Secure Trading System | TradPal Indicator v1.0</p>
        </div>
    """, unsafe_allow_html=True)


def get_user_info(username):
    """Get user information."""
    users = load_users()
    return users.get(username)


def update_user_password(username, old_password, new_password):
    """Update user password."""
    users = load_users()
    
    if username not in users:
        return False, "User not found"
    
    # Verify old password
    user = users[username]
    if not check_password_hash(user['password_hash'], old_password):
        return False, "Invalid current password"
    
    # Update password
    users[username]['password_hash'] = generate_password_hash(new_password)
    users[username]['updated_at'] = datetime.now().isoformat()
    
    save_users(users)
    return True, "Password updated successfully"


def delete_user(username, admin_username):
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


def list_users(admin_username):
    """List all users (admin only)."""
    users = load_users()
    
    if admin_username not in users or users[admin_username]['role'] != 'admin':
        return None, "Admin privileges required"
    
    user_list = []
    for username, info in users.items():
        user_list.append({
            'username': username,
            'role': info['role'],
            'created_at': info['created_at']
        })
    
    return user_list, "Success"
