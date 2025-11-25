import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import face_recognition
import time
import requests
import json
import sys 

def get_settings_path():
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller EXE
        return os.path.join(os.path.dirname(sys.executable), "settings.json")
    return "settings.json"

SETTINGS_FILE = get_settings_path()

# Load settings or defaults
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {"server_url": "http://127.0.0.1:8000"}

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

settings = load_settings()
server_url = settings.get("server_url")

# === SERVER COMMUNICATION ===
def send_login_to_server(user_id, status):
    """
    Sends login/logout request to server and returns username and full_name for popup.
    """
    #server_url = f"http://192.168.1.20:8000/dtr/timeclock?id={user_id}&status={status}"
    
    try:
        response = requests.get(f"{server_url}/dtr/timeclock?id={user_id}&status={status}")
        data = response.json()

        server_username = data.get('username', user_id)
        server_full_name = data.get('full_name', user_id)  # fallback to username if full_name not provided

        # Check if the login status is success
        if data.get('success') == 'login':
            print(f"✅ {server_username} ({server_full_name}) logged in successfully")
        elif data.get('success') == 'logout':
            print(f"✅ {server_username} ({server_full_name}) logged out successfully")
        # Check for 'fail' response and specific message for unregistered users
        elif data.get('success') == 'fail':
            error_message = data.get('message', 'Unknown error')
            print(f"⚠️ Login failed: {error_message}")
            return None, error_message  # Return None and the error message
        else:
            print(f"⚠️ Server response: {data}")

        return server_username, server_full_name
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return None, "Server connection failed"

# === CONFIGURATION ===
FACE_TEMPLATE_FILE = "face_templates.npz"
CAPTURE_FRAMES = 5
RECOGNITION_TOLERANCE = 0.32  # lower = stricter
MIN_FACE_SIZE = 170#120   # too far
MAX_FACE_SIZE = 200#300   # too close

# === Load Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Utility ===
def load_templates(file):
    if os.path.exists(file):
        return dict(np.load(file, allow_pickle=True))
    return {}

def save_templates(templates, file):
    np.savez(file, **templates)

# === MAIN APP ===
class FacialBiometricLoginApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eversoft Facial Biometric Login System")
        self.root.geometry("1200x700")
        self.root.attributes("-fullscreen", True)  # <-- Enable fullscreen automatically
        self.root.configure(bg="#0B132B")

        
        # Optional: bind Escape key to exit fullscreen
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        # Variables
        self.cap = None
        self.mode = None
        self.face_templates = load_templates(FACE_TEMPLATE_FILE)
        self.running = False
        self.capture_buffer = []
        self.logged_in = False
        self.logged_out = False
        self.logged_in_user = None
        self.user_id = None
        self.last_popup_message = None
        self.status_text = ""  # Overlay text on camera

        # === Title (Scrolling) ===
        # self.title_text = "   EVERSOFT FACIAL BIOMETRIC LOGIN SYSTEM   "
        # self.title_index = 0
        # self.title_label = tk.Label(
        #     root,
        #     text=self.title_text,
        #     font=("Arial Black", 30),
        #     fg="#6FFFE9",
        #     bg="#0B132B"
        # )
        # self.title_label.pack(pady=10)
        title_frame = tk.Frame(root, bg="#0B132B")
        title_frame.pack(pady=10)

        # Load and display the logo
        logo_image = Image.open("logo.png")
        logo_image = logo_image.resize((80, 80), Image.Resampling.LANCZOS)  # Updated Pillow resize
        logo_photo = ImageTk.PhotoImage(logo_image)

        logo_label = tk.Label(title_frame, image=logo_photo, bg="#0B132B")
        logo_label.image = logo_photo  # keep a reference
        logo_label.pack(side=tk.LEFT, padx=(0, 10))

        # Title text next to the logos
        self.title_text = "DAHFI FACIAL BIOMETRIC LOGIN SYSTEM"
        self.title_label = tk.Label(
            title_frame,  # <-- attach to title_frame
            text=self.title_text,
            font=("Arial Black", 30),
            fg="#6FFFE9",
            bg="#0B132B"
        )
        self.title_label.pack(side=tk.LEFT)
       # self.animate_title()

        # === Clock ===
        self.time_label = tk.Label(
            root,
            text="",
            font=("Arial", 20, "bold"),
            fg="#6FFFE9",
            bg="#0B132B"
        )
        self.time_label.pack(anchor="ne", padx=20, pady=5)
        self.update_clock()

        # === Main Frame ===
        main_frame = tk.Frame(root, bg="#1C2541")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left - Video Panel
        self.left_frame = tk.Frame(main_frame, bg="#000000", width=800, height=600)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.left_frame.pack_propagate(False)
        self.video_label = tk.Label(self.left_frame, bg="black")
        self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Right - Controls
        self.right_frame = tk.Frame(main_frame, bg="#1C2541", width=350)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Button Style ---
        self.button_style = {
            "font": ("Arial", 13, "bold"),
            "bg": "#3A506B",
            "fg": "#000000",
            "activebackground": "#5BC0BE",
            "activeforeground": "#FFFFFF",
            "relief": "flat",
            "width": 20,
            "height": 2,
            "bd": 0,
            "cursor": "hand2",
        }

        # Buttons
        self.add_button("Register Face", self.register_face)
        self.add_button("Login with Face", lambda: self.start_camera("login"))
        self.add_button("Logout with Face", self.logout_user)
        self.add_button("Delete Registered Face", self.delete_face)
        self.add_button("Stop Camera", self.stop_camera)  # <<< New button
        self.add_button("Settings", self.open_settings)  # Settings button
        # self.add_button("Login with ID Only", self.login_with_id_only)
        # self.add_button("Logout with ID Only", self.logout_with_id_only)

        # --- Message Area ---
        msg_label = tk.Label(
            self.right_frame,
            text="System Logs",
            bg="#1C2541",
            fg="#6FFFE9",
            font=("Arial", 15, "bold")
        )
        msg_label.pack(pady=(20, 5))

        # Scrollable Text Area
        msg_frame = tk.Frame(self.right_frame, bg="#1C2541")
        msg_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(msg_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.message_text = tk.Text(
            msg_frame,
            width=40,
            height=18,
            bg="#0B132B",
            fg="#F8F9FA",
            insertbackground="#6FFFE9",
            font=("Consolas", 11),
            wrap=tk.WORD,
            relief=tk.FLAT,
            yscrollcommand=scrollbar.set
        )
        self.message_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.message_text.yview)

    # === Title scrolling ===
    # def animate_title(self):
    #     display_text = self.title_text[self.title_index:] + self.title_text[:self.title_index]
    #     self.title_label.config(text=display_text)
    #     self.title_index = (self.title_index + 1) % len(self.title_text)
    #     self.root.after(200, self.animate_title)

    # === Clock update ===
    def update_clock(self):
        current_time = time.strftime("%I:%M:%S %p")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_clock)

    # === Button helper ===
    def add_button(self, text, command):
        btn = tk.Button(self.right_frame, text=text, command=command, **self.button_style)
        btn.pack(pady=6)
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#5BC0BE"))
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#3A506B"))

    # === Message display ===
    def add_message(self, msg):
        self.message_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.message_text.see(tk.END)
        self.message_text.update_idletasks()

    # === Camera control ===
    def start_camera(self, mode):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the camera.")
            return

        # Reset states for each new login attempt
        if mode == "login":
            self.logged_in = False  # Reset login state
            self.logged_in_user = None

        self.mode = mode
        self.running = True
        self.capture_buffer = []
        self.status_text = ""
        self.start_time = time.time()  # ⏱️ record start time
        self.add_message(f"Camera started in {mode.upper()} mode. Initializing...")

        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        black_img = Image.fromarray(black_img)
        imgtk = ImageTk.PhotoImage(image=black_img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.add_message("Camera stopped.")
        self.mode = None
        self.status_text = ""

    def ask_password(self):
        pw_window = tk.Toplevel(self.root)
        pw_window.title("Enter Admin Password")
        pw_window.geometry("350x160")
        pw_window.configure(bg="#1C2541")
        pw_window.grab_set()

        tk.Label(
            pw_window,
            text="Enter admin password to continue:",
            bg="#1C2541",
            fg="#6FFFE9",
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        pw_var = tk.StringVar()
        pw_entry = tk.Entry(
            pw_window, textvariable=pw_var, show="*",
            font=("Arial", 12), width=25
        )
        pw_entry.pack(pady=5)

        result = {"password": None}

        def confirm():
            result["password"] = pw_var.get()
            pw_window.destroy()

        tk.Button(
            pw_window, text="OK",
            command=confirm,
            font=("Arial", 10, "bold"),
            bg="#FFFFFF", fg="black",
            width=10
        ).pack(pady=10)

        pw_window.wait_window()
        return result["password"]
    # === Register Face ===
    # def register_face(self):
    #     user_id = simpledialog.askstring("Register Face", "Enter User ID:")
    #     if not user_id:
    #         return
    #     if user_id in self.face_templates:
    #         overwrite = messagebox.askyesno("User Exists", f"{user_id} already exists. Overwrite?")
    #         if not overwrite:
    #             return
    #     self.user_id = user_id
    #     self.start_camera("register")
    def register_face(self):
        # 1. Ask for User ID
        user_id = simpledialog.askstring("Register Face", "Enter User ID:")
        if not user_id:
            return

        # 2. If user exists, confirm overwrite
        if user_id in self.face_templates:
            overwrite = messagebox.askyesno(
                "User Exists",
                f"User '{user_id}' already exists.\nDo you want to overwrite?"
            )
            if not overwrite:
                return

        # 3. Ask for admin password
        password = self.ask_password()
        if password is None:
            messagebox.showwarning("Cancelled", "Registration cancelled.")
            return

        # 4. Validate password
        if password != "admin123":
            messagebox.showerror("Incorrect Password", "The admin password is incorrect!")
            return

        # 5. Access granted → start camera for face registration
        messagebox.showinfo("Access Granted", "Password accepted. Starting face registration...")

        self.user_id = user_id
        self.start_camera("register")

    # === Delete Face ===
    # def delete_face(self):
    #     if not self.face_templates:
    #         messagebox.showinfo("No Data", "No registered faces found.")
    #         return

    #     user_list = list(self.face_templates.keys())
    #     user_to_delete = simpledialog.askstring(
    #         "Delete Face",
    #         f"Registered Users:\n\n{', '.join(user_list)}\n\nEnter User ID to Delete:"
    #     )
    #     if user_to_delete in self.face_templates:
    #         del self.face_templates[user_to_delete]
    #         save_templates(self.face_templates, FACE_TEMPLATE_FILE)
    #         self.add_message(f"Deleted registered face: {user_to_delete}")
    #         messagebox.showinfo("Deleted", f"User '{user_to_delete}' deleted successfully.")
    #     else:
    #         messagebox.showerror("Not Found", f"No user found with ID: {user_to_delete}")
    def delete_face(self):
        # Make sure users exist
        if not self.face_templates:
            messagebox.showinfo("No Data", "No registered users found.")
            return

        # 1. Ask for admin password
        password = self.ask_password()
        if password is None:
            messagebox.showwarning("Cancelled", "Operation cancelled.")
            return

        if password != "admin123":
            messagebox.showerror("Incorrect Password", "Admin password is incorrect!")
            return

        # 2. Password correct → show list of users
        user_list = list(self.face_templates.keys())
        user_to_delete = simpledialog.askstring(
            "Delete User",
            f"Registered Users:\n\n{', '.join(user_list)}\n\nEnter User ID to delete:"
        )

        if not user_to_delete:
            return

        # 3. Delete user if exists
        if user_to_delete in self.face_templates:
            del self.face_templates[user_to_delete]
            save_templates(self.face_templates, FACE_TEMPLATE_FILE)
            self.add_message(f"Deleted user: {user_to_delete}")
            messagebox.showinfo("Deleted", f"User '{user_to_delete}' deleted successfully.")
        else:
            messagebox.showerror("Not Found", f"No user found with ID: {user_to_delete}")

    def login_with_id_only(self):
        user_id = simpledialog.askstring("Login", "Enter User ID:")
        if not user_id:
            return

        # Send login request WITHOUT face
        server_username, server_full_name = send_login_to_server(user_id, "login")

        # Handle server errors
        if server_username is None:
            self.show_popup(f"❌ {server_full_name}", status="error")
            self.add_message(f"⚠️ Login failed: {server_full_name}")
            return

        # Success
        self.show_popup(f"✅ Login Successful for {server_full_name}", status="success")
        self.add_message(f"User '{server_full_name}' logged in successfully.")

    def logout_with_id_only(self):
        user_id = simpledialog.askstring("Logout", "Enter User ID:")
        if not user_id:
            return

        # Send login request WITHOUT face
        server_username, server_full_name = send_login_to_server(user_id, "logout")

        # Handle server errors
        if server_username is None:
            self.show_popup(f"❌ {server_full_name}", status="error")
            self.add_message(f"⚠️ Logout failed: {server_full_name}")
            return

        # Success
        self.show_popup(f"✅ Login Successful for {server_full_name}", status="success")
        self.add_message(f"User '{server_full_name}' logged in successfully.")

    # === Logout User ===
    def logout_user(self):
        self.logged_out = False
    # Add debugging statements
        # print(f"Logged In: {self.logged_in}, Logged In User: {self.logged_in_user}")

        # if not self.logged_in or not self.logged_in_user:
        #     messagebox.showinfo("Logout", "No user currently logged in.")
        #     self.start_camera("logout_preview")  # Show camera anyway
        #     return

        #If logged in, start the logout process
        self.start_camera("logout")

        # Here you can optionally print to debug if we reach this point
        #print(f"Logging out user: {self.logged_in_user}")

    # === Settings Panel ===
    def open_settings(self):
        global server_url, settings

        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings")
        settings_win.configure(bg="#1C2541")
        settings_win.geometry("400x200")
        settings_win.attributes('-topmost', True)

        tk.Label(settings_win, text="Server URL:", font=("Arial", 12, "bold"),
                 bg="#1C2541", fg="#6FFFE9").pack(pady=(20, 5))

        server_url_var = tk.StringVar(value=server_url)
        url_entry = tk.Entry(settings_win, textvariable=server_url_var, font=("Arial", 12), width=40)
        url_entry.pack(pady=5)

        def save_settings_btn():
            global server_url, settings
            server_url = server_url_var.get()
            settings['server_url'] = server_url
            save_settings(settings)
            self.add_message(f"✅ Server URL updated to: {server_url}")
            settings_win.destroy()

        save_btn = tk.Button(settings_win, text="Save", font=("Arial", 12, "bold"),
                             bg="#3A506B", fg="#FFFFFF", activebackground="#5BC0BE",
                             width=15, command=save_settings_btn)
        save_btn.pack(pady=20)

    # === Redesigned Popup ===
    # def show_popup(self, username_fullname, status="success", duration=3000):
    #     popup = tk.Toplevel(self.root)
    #     popup.title("")
    #     popup.configure(bg="#1C2541")
    #     popup.attributes('-topmost', True)
    #     popup.overrideredirect(True)

    #     width, height = 400, 150
    #     screen_width = popup.winfo_screenwidth()
    #     screen_height = popup.winfo_screenheight()
    #     x = (screen_width // 2) - (width // 2)
    #     y = (screen_height // 2) - (height // 2)
    #     popup.geometry(f"{width}x{height}+{x}+{y}")

    #     colors = {"success": "#00FF00", "error": "#FF5555"}
    #     fg_color = colors.get(status, "#6FFFE9")

    #     label = tk.Label(popup, text=username_fullname, font=("Helvetica", 14), fg=fg_color, bg="#1C2541")
    #     label.pack(pady=10)

    #     def close_popup():
    #         popup.destroy()

    #     # Use after() to schedule the popup to close after the given duration
    #     popup.after(duration, close_popup)

    def show_popup(self, username_fullname, status="success", duration=3000):
        # --- PREVENT DUPLICATE POPUP ---
        if self.last_popup_message == (username_fullname, status):
            return  # Prevent same popup from showing again

        # Save this popup as last shown
        self.last_popup_message = (username_fullname, status)

        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("")
        popup.configure(bg="#1C2541")
        popup.attributes('-topmost', True)
        popup.overrideredirect(True)

        # Size + Center
        width, height = 400, 150
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

        # Colors
        colors = {"success": "#00FF00", "error": "#FF5555"}
        fg_color = colors.get(status, "#6FFFE9")

        label = tk.Label(
            popup,
            text=username_fullname,
            font=("Helvetica", 14),
            fg=fg_color,
            bg="#1C2541"
        )
        label.pack(pady=10)

        # Auto close popup
        def close_popup():
            popup.destroy()
            # Reset the last popup after closing so new popups can show
            self.last_popup_message = None

        if status in ("success", "error"):
            self.stop_camera()

        popup.after(duration, close_popup)

        # --- STOP CAMERA ON SUCCESS OR ERROR ---
        # This ensures the camera stops immediately after showing a popup
        

    # === Frame Update ===
    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elapsed = time.time() - getattr(self, 'start_time', 0)
            detect_faces = elapsed >= 2  # start detecting after 2 seconds

            if detect_faces:
                faces = face_recognition.face_locations(rgb_frame)
                encodings = face_recognition.face_encodings(rgb_frame, faces)
            else:
                faces, encodings = [], []
                cv2.putText(frame, "Keep your face within the guide box... Starting soon...",
                            (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # --- CAPTURE ZONE COORDINATES ---
            CAPTURE_X1, CAPTURE_Y1 = 170, 100
            CAPTURE_X2, CAPTURE_Y2 = 470, 380

            # --- BLUR OUTSIDE BOX ---
            blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
            blurred_frame[CAPTURE_Y1:CAPTURE_Y2, CAPTURE_X1:CAPTURE_X2] = frame[CAPTURE_Y1:CAPTURE_Y2, CAPTURE_X1:CAPTURE_X2]
            frame = blurred_frame

            # Draw capture box
            cv2.rectangle(frame, (CAPTURE_X1, CAPTURE_Y1), (CAPTURE_X2, CAPTURE_Y2), (255, 255, 0), 2)
            cv2.putText(frame, "Keep your face within the guide box.", (CAPTURE_X1, CAPTURE_Y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Draw mode/status
            mode_text = f"MODE: {self.mode.upper() if self.mode else 'IDLE'}"
            cv2.putText(frame, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if self.status_text:
                cv2.putText(frame, self.status_text, (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            for (top, right, bottom, left), encoding in zip(faces, encodings):
                # Draw face rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # --- FACE CENTER AND CAPTURE ZONE CHECK ---
                face_center_x = (left + right) // 2
                face_center_y = (top + bottom) // 2

                if not (CAPTURE_X1 <= face_center_x <= CAPTURE_X2 and CAPTURE_Y1 <= face_center_y <= CAPTURE_Y2):
                    self.status_text = "Move your face inside the box"
                    continue  # Skip processing until face is inside zone
                # --- DISTANCE CHECK ---
                face_height = bottom - top
                box_height = CAPTURE_Y2 - CAPTURE_Y1

                # Define min/max size relative to box height
                min_face_size = int(0.5 * box_height)  # face should be at least 50% of box height
                max_face_size = int(0.6 * box_height)  # face should be at most 50% of box height

                if face_height < min_face_size:
                    self.status_text = "Move closer to the camera"
                    self.add_message("Move closer to the camera")
                    continue

                if face_height > max_face_size:
                    self.status_text = "Move back from the camera"
                    self.add_message("Move back from the camera")
                    continue

                # Registration
                # if self.mode == "register":
                #     self.capture_buffer.append(encoding)
                #     self.add_message(f"Captured frame {len(self.capture_buffer)}/{CAPTURE_FRAMES}")
                #     if len(self.capture_buffer) >= CAPTURE_FRAMES:
                #         self.face_templates[self.user_id] = self.capture_buffer.copy()
                #         save_templates(self.face_templates, FACE_TEMPLATE_FILE)
                #         self.show_popup(f"Face Registered: {self.user_id}", status="success")
                #         self.add_message(f"User '{self.user_id}' registered successfully.")
                #         self.stop_camera()

                if self.mode == "register":
                    for (top, right, bottom, left), encoding in zip(faces, encodings):
                        # Calculate face center
                        face_center_x = (left + right) // 2
                        face_center_y = (top + bottom) // 2

                        # Step 1: Align face inside box
                        if not (CAPTURE_X1 <= face_center_x <= CAPTURE_X2 and
                                CAPTURE_Y1 <= face_center_y <= CAPTURE_Y2):
                            self.status_text = "Step 1: Move your face inside the box"
                            break  # wait until user aligns

                        # Step 2: Adjust distance (face size)
                        face_height = bottom - top
                        box_height = CAPTURE_Y2 - CAPTURE_Y1
                        min_face_size = int(0.5 * box_height)
                        max_face_size = int(0.6 * box_height)

                        if face_height < min_face_size:
                            self.status_text = "Step 2: Move closer to the camera"
                            break  # wait until user moves closer
                        if face_height > max_face_size:
                            self.status_text = "Step 2: Move back from the camera"
                            break  # wait until user moves back

                        # Step 3: Capture frame
                        if len(self.capture_buffer) < CAPTURE_FRAMES:
                            self.capture_buffer.append(encoding)
                            frames_captured = len(self.capture_buffer)
                            self.status_text = f"Step 3: Capturing face... Frame {frames_captured}/{CAPTURE_FRAMES}"
                            self.add_message(f"Captured frame {frames_captured}/{CAPTURE_FRAMES}")

                        # Step 4: Save templates once enough frames are captured
                        if len(self.capture_buffer) >= CAPTURE_FRAMES:
                            self.face_templates[self.user_id] = self.capture_buffer.copy()
                            save_templates(self.face_templates, FACE_TEMPLATE_FILE)
                            self.show_popup(f"Face Registered: {self.user_id}", status="success")
                            self.add_message(f"User '{self.user_id}' registered successfully.")
                            self.capture_buffer.clear()
                            self.stop_camera()
                            return

                # Login
                elif self.mode == "login" and not self.logged_in:
                    match_found = False
                    for name, templates in self.face_templates.items():
                        matches = face_recognition.compare_faces(templates, encoding, tolerance=RECOGNITION_TOLERANCE)
                        if matches.count(True) >= 4 and matches[0] == True:#max(1, len(templates)//2):
                            server_username, server_full_name = send_login_to_server(name, "login")
                            if server_username is None:  # Handle fail case (e.g. user not registered)
                                self.status_text = "Login failed"
                                self.add_message("⚠️ User not registered in the system.")
                                self.show_popup("User not registered", status="error")  # Show error popup
                                return
                            self.show_popup(f"✅ Login successful for {server_full_name}", status="success")
                            self.add_message(f"✅ Login successful for {server_full_name}")
                            self.logged_in = True
                            match_found = True
                            self.status_text = f"Logged in: {server_full_name}"  # Instead of name, show full name from server
                            self.root.after(3000, self.stop_camera)
                            return
                    if not match_found:
                        self.status_text = "Face not recognized"
                        self.add_message("⚠️ Face detected but not recognized.")

                # === Logout ===
                elif self.mode == "logout" and not self.logged_out:
                    print('logout')
                    match_found = False
                    # We don't use self.logged_in_user anymore
                    for name, templates in self.face_templates.items():
                        matches = face_recognition.compare_faces(templates, encoding, tolerance=RECOGNITION_TOLERANCE)
                        if matches.count(True) >= 4 and matches[0] == True:# max(1, len(templates)//2):
                            server_username, server_full_name = send_login_to_server(name, "logout")
                            if server_username is None:  # Handle fail case (e.g. user not registered or not logged in)
                                self.status_text = "Logout failed"
                                self.add_message("⚠️ User not registered or not logged in.")
                                self.show_popup("User not registered or not logged in", status="error")  # Show error popup
                                return
                            self.show_popup(f"✅ Logout successful for {server_full_name}", status="success")
                            self.add_message(f"✅ Logout successful for {server_username}")
                            match_found = True
                            self.logged_out = True
                            self.logged_in = False
                            self.status_text = "Logged out successfully"
                            

                            self.root.after(3000, self.stop_camera)
                            return
                    if not match_found:
                        self.status_text = "Face does not match registered user"
                        self.add_message("⚠️ Face detected but does not match any registered user.")

            # Display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(100, self.update_frame)

# === MAIN ===
if __name__ == "__main__":
    root = tk.Tk()
    app = FacialBiometricLoginApp(root)
    root.mainloop()
