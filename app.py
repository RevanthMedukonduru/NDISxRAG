from pathlib import Path
import re
import pandas as pd
import streamlit as st
import requests
from typing import Dict, Any

# Define constants
BASE_URL = "http://web:8000"  # Update this with your backend server's base URL
BASE_DIR = Path(__file__).resolve().parent
HEADERS = {"Content-Type": "application/json"}

# ---------------------------- Components ----------------------------
def login_page():
    """Login Page"""
    st.title("Login Page")
    st.subheader("Enter your login details to proceed")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        data = {"username": username, "password": password}
        response = requests.post(f"{BASE_URL}/rag/login/", data=data, timeout=60)
        response_json = response.json()
        
        if response.status_code == 200:
            st.success("Login successful!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["is_staff"] = response_json.get("is_staff")
            st.session_state["token"] = response_json.get("token")
        else:
            st.error(f"Login failed: {response_json.get('error')}")


def signup_page():
    """Signup Page"""
    st.title("Signup Page")
    st.subheader("Create a new account (requires super key)")

    super_key = st.text_input("Super Key", type="password", key="signup_super_key")
    username = st.text_input("Choose a Username", key="signup_username")
    password = st.text_input("Choose a Password", type="password", key="signup_password")
    is_staff = st.checkbox("Admin Account", key="signup_is_admin")

    if st.button("Signup"):
        data = {"super_key": super_key, "username": username, "password": password, "is_staff": is_staff}
        response = requests.post(f"{BASE_URL}/rag/signup/", data=data, timeout=60)
       
        if response.status_code == 201:
            st.success("Signup successful! You can now log in.")
        else:
            st.error(f"Signup failed: {response.json().get('error')}")

def user_management_ui():
    """
    User Management UI for Admins
    """
    st.title("User Management")

    # Ensure session state variables are initialized
    if "show_users" not in st.session_state:
        st.session_state.show_users = False
    if "original_statuses" not in st.session_state:
        st.session_state.original_statuses = {}

    # Button to fetch and display all users
    if st.button("Show All Users"):
        headers = {
            "Authorization": f"Token {st.session_state.get('token')}"
        }
        response = requests.get(f"{BASE_URL}/rag/update-staff/", headers=headers, timeout=60)

        if response.status_code == 200:
            users = response.json()
            st.session_state.show_users = True
            st.session_state.users = users  # Store user data
            st.session_state.original_statuses = {user['id']: user['is_staff'] for user in users}

        elif response.status_code == 403:
            st.error(response.json().get("error"))
        else:
            st.error("Failed to retrieve user data. Please try again.")

    if st.session_state.show_users:
        with st.expander("All Users", expanded=True):
            st.write("List of all users:")
            user_status_updates = {}

            # Use a form to batch updates
            with st.form("update_staff_form"):
                for user in st.session_state.get('users', []):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**User ID:** {user['id']} | **Username:** {user['username']}")

                    with col2:
                        # Display checkbox using the initial state from session
                        is_staff_toggle = st.checkbox(
                            "Is Staff",
                            value=st.session_state.original_statuses.get(user['id'], False),
                            key=f"staff_toggle_{user['id']}"
                        )
                        user_status_updates[user['id']] = is_staff_toggle

                # Add a submit button
                submitted = st.form_submit_button("Update Staff Status")

                if submitted:
                    headers = {
                        "Authorization": f"Token {st.session_state.get('token')}"
                    }
                    for user_id, new_status in user_status_updates.items():
                        # Only send requests if the status has changed
                        if new_status != st.session_state.original_statuses[user_id]:
                            payload = {"id": user_id, "is_staff": new_status, "username": st.session_state.get("username")}
                            post_response = requests.post(f"{BASE_URL}/rag/update-staff/", json=payload, headers=headers, timeout=60)

                            # Handle the response safely
                            try:
                                if post_response.status_code == 200:
                                    st.success(f"Successfully updated staff status for User ID {user_id}.")
                                    st.session_state.original_statuses[user_id] = new_status  # Update session state
                                else:
                                    response_json = post_response.json()
                                    st.error(f"Failed to update User ID {user_id}: {response_json.get('error', 'Unknown error')}")
                            except requests.exceptions.JSONDecodeError:
                                st.error(f"Failed to update User ID {user_id}: Invalid response from the server.")

def admin_dashboard_page():
    """
    Dashboard for uploading files and listing previously uploaded files.
    """
    st.write(f"## Welcome, {st.session_state.get('username')}!")

    # Update Header with bearer token
    headers = {
        "Authorization": f"Token {st.session_state.get('token')}"
    }
    
    # Fetch the list of previously uploaded files
    uploaded_files_response = requests.get(f"{BASE_URL}/rag/list-files/", params={"username": st.session_state["username"]}, headers=headers, timeout=60)
            
    # Check if the user is an admin
    if st.session_state["logged_in"]:
        if st.session_state.get("is_staff"):
            # Display the user management UI
            user_management_ui()
            
            # -- File Upload UI --
            uploaded_file = st.file_uploader("Choose a file (PDF or Excel)", type=["pdf", "xlsx", "xls"])

            if uploaded_file is not None:
                # Show file name and type in UI
                st.write(f"**Selected file**: {uploaded_file.name}")
                st.write(f"**File type**: {uploaded_file.type}")

            if st.button("Upload"):
                if uploaded_file:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    data = {
                        "username": st.session_state["username"]  # or some auth token
                    }

                    # Send file to the Django endpoint
                    endpoint_url = f"{BASE_URL}/rag/upload-file/"
                    response = requests.post(endpoint_url, files=files, data=data, headers=headers, timeout=60)

                    if response.status_code == 201:
                        st.success("File uploaded successfully! Background processing started.")
                    else:
                        st.error("Failed to upload the file.")
                else:
                    st.warning("No file selected. Please choose a file to upload.")

            # -- List of Previously Uploaded Files --
            st.write("---")
            st.subheader("Previously Uploaded Files")
            if uploaded_files_response.status_code == 200:
                files_data = uploaded_files_response.json()  # Expecting a list of dicts with 'filename' and 'file_type'
                if files_data:
                    for item in files_data:
                        # Convert dict to markdown bullet points
                        for key, value in item.items():
                            st.write(f"- **{key}**: {value}")
                        st.write("---")
                else:
                    st.write("No files uploaded yet.")
            else:
                st.error("Failed to fetch file list.")
        else:
            # Get the recent file:
            uploaded_files = uploaded_files_response.json()[0]
            
            if len(uploaded_files) == 0:
                st.write("No files uploaded yet.")
                return

            st.write("**You are currently chatting with below file (File Info):**", uploaded_files)
            
            # Read all sheets at once into a dictionary of DataFrames
            overall_path = f"{BASE_DIR}{uploaded_files['file']}"
            all_sheets = pd.read_excel(overall_path, sheet_name=None)  # None reads all sheets

            # Display each sheet
            for sheet_idx, (sheet_name, sheet_df) in enumerate(all_sheets.items()):
                st.write(f"### Sheet {sheet_idx + 1}: {sheet_name}")
                st.dataframe(data=sheet_df, height=1000, width=1500)

def clean_markdown(text: str) -> str:
    """
    Clean and normalize markdown formatting.
    
    Args:
        text (str): Raw markdown text
        
    Returns:
        str: Cleaned markdown text
    """
    if not text:
        return text
        
    text = text.strip().strip('"').strip("'") # Remove leading and trailing quotes if they exist
    text = text.strip()  # Remove excess whitespace
    text = text.replace("\n", "\n\n")  # Convert single newlines to paragraphs
    text = text.replace("\\n", "\n")  # If backend sends escaped newlines
    text = text.replace("\t", "    ")  # Replace tabs with spaces
    return text.strip()

def format_message(msg: Dict) -> str:
    """
    Format a message for display, adding appropriate styling.
    
    Args:
        msg (Dict): Message dictionary containing role and content
        
    Returns:
        str: Formatted message content
    """
    content = msg["content"]
    
    # Add role-specific formatting if needed
    if msg["role"] == "assistant":
        # Ensure code blocks are properly formatted
        content = re.sub(r'(?s)```(\w+)?\n(.*?)```', r'```\1\n\2\n```', content)
        
    return clean_markdown(content)

def chat_page():
    """
    A Streamlit page that:
      - Displays a chat interface with user/assistant messages
      - Captures user input
      - Sends conversation history to Django backend
      - Streams and displays the assistant's response with proper markdown
    """
    st.title("NDIS Pricing AI Assistant")
    
    # Initialize conversation history only if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Get API token from session
    token = st.session_state.get('token')
    if not token:
        st.error("Please log in first")
        return
        
    # Set up request headers
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }

    # Custom CSS for better markdown display
    st.markdown("""
        <style>
            .stMarkdown {
                line-height: 1.5;
                margin-bottom: 1rem;
            }
            .stMarkdown p {
                margin-bottom: 1rem;
            }
            .stMarkdown ul, .stMarkdown ol {
                margin-bottom: 1rem;
                padding-left: 2rem;
            }
            .stMarkdown li {
                margin-bottom: 0.5rem;
            }
            .stMarkdown pre {
                margin-bottom: 1rem;
                padding: 1rem;
                background-color: #f6f8fa;
                border-radius: 6px;
            }
            .stMarkdown code {
                padding: 0.2em 0.4em;
                background-color: #f6f8fa;
                border-radius: 3px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            formatted_content = format_message(msg)
            st.markdown(formatted_content)
            if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
                with st.expander("Sources", expanded=False):
                    st.markdown(f"- {msg['citations']}")

    # Handle new user input
    if prompt := st.chat_input("Example: What is the price for washing dishes in south australia?"):
        # Check if this is a new message to prevent duplication
        if not any(msg["role"] == "user" and msg["content"] == prompt 
                  for msg in st.session_state.messages[-2:]):
            
            # Add and display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Your AI Assistant is analysing...")
                
                try:
                    # Send request to Django backend using BASE_URL
                    response = requests.post(
                        f"{BASE_URL}/rag/chat/",
                        json={
                            "chat_history": st.session_state.messages,
                            "query": prompt
                        },
                        headers=headers,
                        timeout=120
                    )
                    response.raise_for_status()
                    
                    # Process response
                    cleaned_response = clean_markdown(response.json().get("response", ""))
                    citations = response.json().get("citations", "")
                    
                    # Check if this response is already present
                    if not any(msg["role"] == "assistant" and msg["content"] == cleaned_response 
                             for msg in st.session_state.messages[-2:]):
                        
                        # Store assistant response with citations
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": cleaned_response,
                            "citations": citations
                        })
                    
                    # Update the placeholder with the response
                    message_placeholder.markdown(cleaned_response)
                    
                except requests.RequestException as e:
                    error_msg = f"**Error:** Failed to get response. {str(e)}"
                    message_placeholder.markdown(error_msg)
                    if not any(msg["role"] == "assistant" and msg["content"] == error_msg 
                             for msg in st.session_state.messages[-2:]):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    # Add a clear chat button
    if st.sidebar.button("Clear Chat", help="Click to clear the chat history"):
        st.session_state.messages = []
        st.rerun()
    
# ----------------------------
# Main App
# ----------------------------
def main():
    """
    Main App
    """
    # Ensure session state keys exist
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Login", "Signup", "Dashboard", "Chat"])
    
    # Add a logout button to the sidebar (Place it at the bottom left)
    if st.session_state["logged_in"]:
        if st.sidebar.button("Logout", key="logout_button", help="Click to log out"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.session_state["is_staff"] = False
            st.session_state["token"] = None
            st.session_state["show_users"] = False
            st.session_state["original_statuses"] = {}

    # Route to the selected page
    if page == "Login":
        login_page()
    elif page == "Signup":
        signup_page()
    elif page == "Dashboard":
        if st.session_state["logged_in"]:
            admin_dashboard_page()
        else:
            st.warning("Please log in to access the Dashboard.")
    elif page == "Chat":
        if st.session_state["logged_in"]:
            chat_page()
        else:
            st.warning("Please log in to access the Chat page.")

# Run the app
if __name__ == "__main__":
    main()
