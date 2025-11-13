import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import face_recognition
import time
import requests

# === SERVER COMMUNICATION ===
def send_login_to_server(user_id, status):
    """
    Sends login/logout request to server and returns username and full_name for popup.
    """
    server_url = f"http://192.168.1.20:8000/dtr/timeclock?id={user_id}&status={status}"
    try:
        response = requests.get(server_url)
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
RECOGNITION_TOLERANCE = 0.4  # lower = stricter

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
        self.root.configure(bg="#0B132B")

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
        self.status_text = ""  # Overlay text on camera

        # === Title (Scrolling) ===
        self.title_text = "   EVERSOFT FACIAL BIOMETRIC LOGIN SYSTEM   "
        self.title_index = 0
        self.title_label = tk.Label(
            root,
            text=self.title_text,
            font=("Arial Black", 30),
            fg="#6FFFE9",
            bg="#0B132B"
        )
        self.title_label.pack(pady=10)
        self.animate_title()

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
            "fg": "#FFFFFF",
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
    def animate_title(self):
        display_text = self.title_text[self.title_index:] + self.title_text[:self.title_index]
        self.title_label.config(text=display_text)
        self.title_index = (self.title_index + 1) % len(self.title_text)
        self.root.after(200, self.animate_title)

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


    # === Register Face ===
    def register_face(self):
        user_id = simpledialog.askstring("Register Face", "Enter User ID:")
        if not user_id:
            return
        if user_id in self.face_templates:
            overwrite = messagebox.askyesno("User Exists", f"{user_id} already exists. Overwrite?")
            if not overwrite:
                return
        self.user_id = user_id
        self.start_camera("register")

    # === Delete Face ===
    def delete_face(self):
        if not self.face_templates:
            messagebox.showinfo("No Data", "No registered faces found.")
            return

        user_list = list(self.face_templates.keys())
        user_to_delete = simpledialog.askstring(
            "Delete Face",
            f"Registered Users:\n\n{', '.join(user_list)}\n\nEnter User ID to Delete:"
        )
        if user_to_delete in self.face_templates:
            del self.face_templates[user_to_delete]
            save_templates(self.face_templates, FACE_TEMPLATE_FILE)
            self.add_message(f"Deleted registered face: {user_to_delete}")
            messagebox.showinfo("Deleted", f"User '{user_to_delete}' deleted successfully.")
        else:
            messagebox.showerror("Not Found", f"No user found with ID: {user_to_delete}")

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

    # === Redesigned Popup ===
    def show_popup(self, username_fullname, status="success", duration=3000):
        popup = tk.Toplevel(self.root)
        popup.title("")
        popup.configure(bg="#1C2541")
        popup.attributes('-topmost', True)
        popup.overrideredirect(True)

        width, height = 400, 150
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

        colors = {"success": "#00FF00", "error": "#FF5555"}
        fg_color = colors.get(status, "#6FFFE9")

        label = tk.Label(popup, text=username_fullname, font=("Helvetica", 14), fg=fg_color, bg="#1C2541")
        label.pack(pady=10)

        def close_popup():
            popup.destroy()

        # Use after() to schedule the popup to close after the given duration
        popup.after(duration, close_popup)

    # === Frame Update ===
    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elapsed = time.time() - getattr(self, 'start_time', 0)
            detect_faces = elapsed >= 2  # ✅ start detecting after 3 seconds

            if detect_faces:
                faces = face_recognition.face_locations(rgb_frame)
                encodings = face_recognition.face_encodings(rgb_frame, faces)
            else:
                faces, encodings = [], []
                cv2.putText(frame, "Adjust your face... Starting soon...",
                            (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw mode/status
            mode_text = f"MODE: {self.mode.upper() if self.mode else 'IDLE'}"
            cv2.putText(frame, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if self.status_text:
                cv2.putText(frame, self.status_text, (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            for (top, right, bottom, left), encoding in zip(faces, encodings):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Registration
                if self.mode == "register":
                    self.capture_buffer.append(encoding)
                    self.add_message(f"Captured frame {len(self.capture_buffer)}/{CAPTURE_FRAMES}")
                    if len(self.capture_buffer) >= CAPTURE_FRAMES:
                        self.face_templates[self.user_id] = self.capture_buffer.copy()
                        save_templates(self.face_templates, FACE_TEMPLATE_FILE)
                        self.show_popup(f"Face Registered: {self.user_id}", status="success")
                        self.add_message(f"User '{self.user_id}' registered successfully.")
                        self.stop_camera()

                # Login
                elif self.mode == "login" and not self.logged_in:
                    match_found = False
                    for name, templates in self.face_templates.items():
                        matches = face_recognition.compare_faces(templates, encoding, tolerance=RECOGNITION_TOLERANCE)
                        if matches.count(True) >= max(1, len(templates)//2):
                            server_username, server_full_name = send_login_to_server(name, "login")
                            if server_username is None:  # Handle fail case (e.g. user not registered)
                                self.status_text = "Login failed"
                                self.add_message("⚠️ User not registered in the system.")
                                self.show_popup("User not registered", status="error")  # Show error popup
                                break
                            self.show_popup(f"✅ Login successful for {server_full_name} - {server_username}", status="success")
                            self.add_message(f"✅ Login successful for {server_full_name} - {server_username}")
                            self.logged_in = True
                            match_found = True
                            self.status_text = f"Logged in: {server_full_name}"  # Instead of name, show full name from server
                            self.root.after(3000, self.stop_camera)
                            break
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
                        if matches.count(True) >= max(1, len(templates)//2):
                            server_username, server_full_name = send_login_to_server(name, "logout")
                            if server_username is None:  # Handle fail case (e.g. user not registered or not logged in)
                                self.status_text = "Logout failed"
                                self.add_message("⚠️ User not registered or not logged in.")
                                self.show_popup("User not registered or not logged in", status="error")  # Show error popup
                                break
                            self.show_popup(f"✅ Logout successful for {server_full_name} - {server_username}", status="success")
                            self.add_message(f"✅ Logout successful for {server_username}")
                            match_found = True
                            self.logged_out = True
                            self.logged_in = False
                            self.status_text = "Logged out successfully"
                            

                            self.root.after(3000, self.stop_camera)
                            break
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
