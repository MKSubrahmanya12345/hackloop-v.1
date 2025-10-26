import streamlit as st
import requests
import base64
from PIL import Image
import io

# --- Configuration ---
# This must match the URL of your running main.py backend
BACKEND_API_URL = "http://127.0.0.1:8000/api/identify"
# ---------------------

st.set_page_config(layout="centered", page_title="Attendance System")
st.title("Step 1: Attendance Enrollment")
st.write("Stand in front of the camera and click 'Mark Attendance'.")

# 1. Use Streamlit's camera_input to show the webcam feed
img_file_buffer = st.camera_input("Camera Feed")

if img_file_buffer is not None:
    # 2. When the user clicks the "Take Photo" button...
    
    # Read the image bytes
    img_bytes = img_file_buffer.getvalue()
    
    # Convert image bytes to a base64 string
    # This is exactly what the backend expects
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Show a spinner while we process
    with st.spinner("Analyzing face..."):
        try:
            # 3. Send the API request to your backend
            payload = {"image_base64": img_base64}
            response = requests.post(BACKEND_API_URL, json=payload)
            
            # 4. Handle the response from the backend
            if response.status_code == 200:
                # SUCCESS!
                data = response.json()
                usn = data.get("usn")
                name = data.get("name")
                distance = data.get("distance")
                
                st.success(f"Welcome, {name} (USN: {usn})!")
                st.write(f"Match confidence (distance): {distance:.4f}")
                st.balloons()
                
            elif response.status_code == 404:
                # FAILURE (Not recognized)
                data = response.json()
                detail = data.get("detail")
                st.error(f"Attendance Failed: {detail}")
                
            else:
                # Other server error
                st.error(f"An error occurred. Status Code: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            # Failed to connect to the backend
            st.error("Connection Error: Could not connect to the backend.")
            st.write("Is your 'main.py' server running in the other terminal?")
            st.write(f"Error details: {e}")