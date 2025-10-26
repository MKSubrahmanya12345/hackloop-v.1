import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import requests
import base64
import time
from deepface import DeepFace
from deepface.modules import verification as dst
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import threading  # To run FastAPI in the background
import json

# --- Configuration ---
VIDEO_SOURCE = "http://192.168.1.37:8080/video" # Confirmed phone IP cam URL
AUTH_API_URL = "http://127.0.0.1:8000/api/identify" # Cam 1 API for re-acquisition
MODEL_NAME = "ArcFace"
HANDOFF_THRESHOLD = 0.9  # How close the face must be during hand-off
REACQUIRE_THRESHOLD = 0.9 # Stricter threshold for re-acquiring a lost tag
REVERIFY_THRESHOLD = 0.9 # *Relaxed* strict check for already-tagged people

# --- State Dictionaries (Thread-Safe) ---
lock = threading.Lock()
known_tags = {}
cooldowns = {}
student_to_find = {}

# --- Cooldowns ---
REACQUIRE_COOLDOWN = 150 # 5 seconds (150 frames @ 30fps)
REVERIFY_COOLDOWN = 300 # 10 seconds (300 frames @ 30fps)


# --- Part 1: The API Server for Camera 2 ---
cam2_app = FastAPI(title="Camera 2: Tracker API")

class HandoffPayload(BaseModel):
    usn: str
    live_embedding: list # A list of floats

@cam2_app.post("/api/initiate_tag")
async def initiate_tag(payload: HandoffPayload):
    global student_to_find
    with lock:
        # Check if we are already trying to find someone, maybe overwrite or queue?
        # For simplicity now, just overwrite.
        student_to_find = {"usn": payload.usn, "embedding": payload.live_embedding}
    print(f"[API_CAM2_INFO] Received hand-off command for {payload.usn}")
    return {"status": "handoff_initiated"}

def start_api_server():
    print("[INFO] Starting Camera 2 API server on port 8001...")
    uvicorn.run(cam2_app, host="0.0.0.0", port=8001, log_level="warning")


# --- Part 2: The OpenCV Tracker Logic ---
def run_tracker():
    global student_to_find, known_tags, cooldowns

    # Load models
    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()

    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_CENTER, text_scale=0.5, text_thickness=1
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {VIDEO_SOURCE}")
        print(f"cv2.VideoCapture error details (if any): {cap.getBackendName()}")
        return

    print("[INFO] Tracker is running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended or frame could not be read. Attempting reconnect or exiting.")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                print("[ERROR] Reconnect failed. Exiting.")
                break
            else:
                print("[INFO] Reconnected to video source.")
                continue

        results = model(frame, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = tracker.update_with_detections(detections)

        labels = [] # Final labels for this frame

        # Decrement cooldowns first
        with lock:
            current_cooldown_keys = list(cooldowns.keys()) # Copy keys to avoid modifying dict during iteration
        for tracker_id in current_cooldown_keys:
             with lock:
                 if tracker_id in cooldowns: # Check if key still exists (might be removed by other logic)
                     cooldowns[tracker_id] -= 1
                     if cooldowns[tracker_id] <= 0:
                         cooldowns.pop(tracker_id)


        for detection_idx, tracker_id in enumerate(tracked_detections.tracker_id):

            bbox = tracked_detections.xyxy[detection_idx]
            x1, y1, x2, y2 = map(int, bbox)

            # --- Initialize label for this track ---
            current_label = "Unknown" # Start assuming unknown
            skip_further_checks = False # Flag to skip logic if handoff succeeds

            # --- LOGIC 1: HAND-OFF (Top Priority) ---
            handoff_usn_to_find = None
            handoff_embedding_to_find = None
            is_eligible_for_handoff = False

            with lock:
                 if student_to_find and tracker_id not in known_tags and tracker_id not in cooldowns: # Check cooldown too
                     is_eligible_for_handoff = True
                     handoff_usn_to_find = student_to_find["usn"]
                     handoff_embedding_to_find = student_to_find["embedding"]

            if is_eligible_for_handoff:
                print(f"[DEBUG_HANDOFF] Entered handoff block for tracker_id {tracker_id}, searching for {handoff_usn_to_find}")

                crop_margin = 10
                face_crop = frame[max(0, y1 - crop_margin):min(frame.shape[0], y2 + crop_margin),
                                  max(0, x1 - crop_margin):min(frame.shape[1], x2 + crop_margin)]

                if face_crop.size > 0:
                    try:
                        live_rep = DeepFace.represent(face_crop, model_name=MODEL_NAME, enforce_detection=False, detector_backend='skip')
                        live_embedding = live_rep[0]["embedding"]

                        distance = dst.find_cosine_distance(
                            np.array(live_embedding, dtype=np.float32),
                            np.array(handoff_embedding_to_find, dtype=np.float32)
                        )

                        print(f"[DEBUG_HANDOFF]  -> Distance for {tracker_id}: {distance:.4f}")

                        if distance < HANDOFF_THRESHOLD:
                            print(f"[SUCCESS_HANDOFF] Tagged {handoff_usn_to_find} to tracker_id {tracker_id}")
                            with lock:
                                known_tags[tracker_id] = handoff_usn_to_find
                                student_to_find = {} # Clear handoff request globally
                                cooldowns[tracker_id] = REVERIFY_COOLDOWN # Apply cooldown immediately
                            current_label = handoff_usn_to_find # Update label immediately
                            skip_further_checks = True # --- ADDED: Skip other logic for this ID in this frame ---

                        else:
                            print(f"[DEBUG_HANDOFF]  -> Match failed threshold ({distance:.4f} > {HANDOFF_THRESHOLD})")
                            # Put on short reacquire cooldown if handoff fails? Maybe not needed yet.
                            # with lock:
                            #     cooldowns[tracker_id] = REACQUIRE_COOLDOWN // 3 # Short cooldown

                    except Exception as e:
                         print(f"[ERROR_HANDOFF]  -> Exception during represent/distance for {tracker_id}: {e}")
                else:
                    print(f"[DEBUG_HANDOFF]  -> Face crop size is zero for {tracker_id}")


            # --- Skip remaining checks if handoff just succeeded ---
            if skip_further_checks:
                labels.append(current_label)
                continue # Go to the next tracker_id in the loop

            # --- Determine state *after* attempting handoff ---
            needs_reacquire_check = False
            needs_reverify_check = False
            label_from_state = "Unknown"

            with lock:
                if tracker_id in known_tags:
                    label_from_state = known_tags[tracker_id]
                    if tracker_id not in cooldowns:
                         needs_reverify_check = True
                elif tracker_id not in cooldowns:
                    needs_reacquire_check = True

            current_label = label_from_state # Use state label if handoff didn't apply


            # --- LOGIC 2: RE-VERIFICATION (Security Check) ---
            if needs_reverify_check:
                # print(f"[DEBUG_REVERIFY] Checking tagged {tracker_id} ({current_label})")
                crop_margin = 10
                face_crop = frame[max(0, y1 - crop_margin):min(frame.shape[0], y2 + crop_margin),
                                  max(0, x1 - crop_margin):min(frame.shape[1], x2 + crop_margin)]
                if face_crop.size > 0:
                    try:
                        live_rep = DeepFace.represent(face_crop, model_name=MODEL_NAME, enforce_detection=False, detector_backend='skip')
                        live_embedding = live_rep[0]["embedding"]

                        enrolled_embeddings = face_database.get(current_label, [])
                        best_dist = float('inf')
                        if not enrolled_embeddings:
                             print(f"[WARNING] No enrolled embeddings for USN {current_label}. Untagging.")
                             with lock:
                                 known_tags.pop(tracker_id, None)
                                 cooldowns[tracker_id] = REACQUIRE_COOLDOWN # Needs re-acquisition
                             current_label = "Unknown" # Update label for annotation
                        else:
                            for enrolled_emb in enrolled_embeddings:
                                dist = dst.find_cosine_distance(
                                    np.array(live_embedding, dtype=np.float32),
                                    np.array(enrolled_emb, dtype=np.float32)
                                )
                                if dist < best_dist:
                                    best_dist = dist

                            if best_dist > REVERIFY_THRESHOLD:
                                print(f"[WARNING] Re-verify failed! {tracker_id} is NOT {current_label} (dist: {best_dist:.4f}). Untagging.")
                                with lock:
                                    known_tags.pop(tracker_id, None)
                                current_label = "Suspicious"
                                with lock:
                                     cooldowns[tracker_id] = REACQUIRE_COOLDOWN
                            else:
                                # Re-verify passed
                                # print(f"[DEBUG_REVERIFY] Passed {tracker_id} ({current_label}) dist: {best_dist:.4f}")
                                with lock:
                                     cooldowns[tracker_id] = REVERIFY_COOLDOWN

                    except Exception as e:
                        print(f"[ERROR_REVERIFY] Exception for {tracker_id} ({current_label}): {e}")
                        with lock:
                            cooldowns[tracker_id] = REVERIFY_COOLDOWN # Apply cooldown on error
                        # Keep the current tag on error, don't untag
                        pass
                else:
                    # print(f"[DEBUG_REVERIFY] Face crop zero for {tracker_id} ({current_label})")
                    with lock:
                        cooldowns[tracker_id] = REVERIFY_COOLDOWN # Apply cooldown if crop empty


            # --- LOGIC 3: RE-ACQUISITION (Find Lost Students) ---
            elif needs_reacquire_check:
                # print(f"[DEBUG_REACQUIRE] Checking untagged {tracker_id}")
                crop_margin = 10
                face_crop = frame[max(0, y1 - crop_margin):min(frame.shape[0], y2 + crop_margin),
                                  max(0, x1 - crop_margin):min(frame.shape[1], x2 + crop_margin)]
                if face_crop.size > 0:
                    try:
                        _, buffer = cv2.imencode(".jpg", face_crop)
                        if buffer is None:
                            raise ValueError("cv2.imencode failed")
                        face_base64 = base64.b64encode(buffer).decode('utf-8')

                        payload = {"image_base64": face_base64}
                        response = requests.post(AUTH_API_URL, json=payload, timeout=0.3)

                        if response.status_code == 200:
                            data = response.json()
                            if data.get("distance", 1.0) < REACQUIRE_THRESHOLD:
                                usn = data.get("usn")
                                print(f"[RE-ACQUIRED] Found {usn}, tagging {tracker_id}")
                                with lock:
                                    known_tags[tracker_id] = usn
                                current_label = usn
                                with lock:
                                    cooldowns[tracker_id] = REVERIFY_COOLDOWN
                            else:
                                with lock:
                                    cooldowns[tracker_id] = REACQUIRE_COOLDOWN

                        elif response.status_code == 404:
                             with lock:
                                 cooldowns[tracker_id] = REACQUIRE_COOLDOWN
                        else:
                             print(f"[ERROR_REACQUIRE] Auth API status {response.status_code}")
                             with lock:
                                 cooldowns[tracker_id] = REACQUIRE_COOLDOWN

                    except requests.exceptions.Timeout:
                        with lock:
                            cooldowns[tracker_id] = REACQUIRE_COOLDOWN
                        pass
                    except Exception as e:
                        print(f"[ERROR_REACQUIRE] Exception for {tracker_id}: {e}")
                        with lock:
                            cooldowns[tracker_id] = REACQUIRE_COOLDOWN
                        pass
                else:
                    with lock:
                        cooldowns[tracker_id] = REACQUIRE_COoldown # Apply cooldown if crop empty

            # --- Append the final label determined by the logic ---
            labels.append(current_label)

        # --- Annotation ---
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)

        # Safety check for label count mismatch
        if len(labels) != len(tracked_detections.tracker_id):
             print(f"[ERROR] Label count ({len(labels)}) != detections ({len(tracked_detections.tracker_id)}). Using safe labels.")
             safe_labels = []
             with lock: # Access known_tags safely
                 for tid in tracked_detections.tracker_id:
                     safe_labels.append(known_tags.get(tid, "Unknown"))
             labels = safe_labels # Overwrite potentially corrupted labels

        # Annotate only if counts match after safety check
        if len(labels) == len(tracked_detections.tracker_id):
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=tracked_detections, labels=labels
            )
        else:
             print(f"[CRITICAL ERROR] Label mismatch persisted. Skipping annotation.")


        cv2.imshow("Camera 2 Tracker (Smart Ecosystem)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracker stopped.")

# --- Part 3: Main Execution ---
if __name__ == "__main__":
    face_database = {} # Make global for run_tracker
    try:
        with open("face_database.json", 'r') as f:
            face_database = json.load(f)
        print(f"[INFO_CAM2] Loaded master DB with {len(face_database)} students for re-verification.")
    except FileNotFoundError:
         print(f"[FATAL_CAM2] face_database.json not found.")
    except json.JSONDecodeError:
         print(f"[FATAL_CAM2] Error decoding face_database.json.")
    except Exception as e:
        print(f"[FATAL_CAM2] Could not load face_database.json: {e}")

    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()

    time.sleep(2) # Give server time to start

    run_tracker()

