import streamlit as st
import json
import os

USER_DB = "users.json"

st.set_page_config(
    page_title="BrandPulse AI | Sentiment Intelligence", 
    page_icon="https://cdn-icons-png.flaticon.com/512/2103/2103633.png", 
    layout="wide"
)

def load_data():
    """Loads data and automatically repairs the file structure if needed."""
    # UPDATE THESE TWO STRINGS BELOW
    default_structure = {"users": {"hetu": "ilovemoon"}, "system_active": True}
    
    if not os.path.exists(USER_DB):
        return default_structure
    try:
        with open(USER_DB, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "users" not in data:
            save_data(default_structure)
            return default_structure
        return data
    except (json.JSONDecodeError, KeyError):
        save_data(default_structure)
        return default_structure

def save_data(data):
    with open(USER_DB, "w") as f:
        json.dump(data, f, indent=4)

def check_password():
    data = load_data()
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["is_admin"] = False

    # MASTER KILL SWITCH: Blocks non-admins if system is inactive
    if not data.get("system_active", True) and not st.session_state.get("is_admin"):
        st.error("🚫 System is currently offline for maintenance. Contact Administrator.")
        st.stop()

    if not st.session_state["authenticated"]:
        render_login_ui()
        return False
    return True

def render_login_ui():
    st.markdown("<h1 style='text-align: center;'>🔐 BrandPulse AI | Sentiment Intelligence </h1>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        u_input = st.text_input("Username")
        p_input = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            data = load_data()
            users = data["users"]
            if u_input in users and users[u_input] == p_input:
                st.session_state["authenticated"] = True
                st.session_state["is_admin"] = (u_input == "hetu")
                st.rerun()
            else:
                st.error("Invalid Username or Password")

def render_admin_tab():
    st.subheader("🛡️ Enterprise Control Center")
    data = load_data()
    
    # 1. Kill Switch
    st.write("### System Status")
    is_active = data.get("system_active", True)
    new_status = st.toggle("Global App Access", value=is_active, help="Turning this off blocks all non-admin users.")
    if new_status != is_active:
        data["system_active"] = new_status
        save_data(data)
        st.rerun()

    st.divider()

    # 2. User Management
    st.write("### User Access Control")
    with st.expander("➕ Authorize New User"):
        c1, c2 = st.columns(2)
        new_u = c1.text_input("New Username")
        new_p = c2.text_input("New Password", type="password")
        if st.button("Create Access"):
            if new_u and new_p:
                data["users"][new_u] = new_p
                save_data(data)
                st.success(f"Access granted to {new_u}")
                st.rerun()

    # 3. List and Delete Users
    for u in list(data["users"].keys()):
        if u == "admin": continue
        c_name, c_btn = st.columns([3, 1])
        c_name.write(f"👤 **{u}**")
        if c_btn.button("Revoke", key=f"del_{u}", type="secondary"):
            del data["users"][u]
            save_data(data)
            st.rerun()
            
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()