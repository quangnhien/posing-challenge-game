import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import random
import helper
import os
import numpy as np

import compare_keypoints
# --- Pose setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

# --- Camera ---
cap = cv2.VideoCapture(0)

# --- Create main window ---
root = tk.Tk()
root.title("Posing Game")
root.geometry("1500x900")

# --- Left side: Guide + Button ---
left_frame = tk.Frame(root, width=600, bg="white")
left_frame.pack(side=tk.LEFT, fill=tk.Y)
left_frame.pack_propagate(False)


guide_label = tk.Label(
    left_frame,
    text="🎮 Game Guide:\n\n1. Follow the pose\n2. Hold for 5 seconds\n3. Score will be shown!",
    justify=tk.LEFT,
    bg="gray",
    font=("Helvetica", 12),
    padx=10,
    pady=10
)
guide_label.pack(pady=10)

# Number of images input
tk.Label(left_frame, text="Number of images:", bg="white").pack()
image_count_var = tk.IntVar(value=3)
image_count_spinbox = tk.Spinbox(left_frame, from_=1, to=20, textvariable=image_count_var)
image_count_spinbox.pack(pady=5)

# Pose image display
pose_image_label = tk.Label(left_frame, bg="white")
pose_image_label.pack(pady=10)

# Load pose images (use your actual image folder here)
pose_images = []
pose_image_folder = "./funny"  # Replace with your path

if os.path.exists(pose_image_folder):
    for file in os.listdir(pose_image_folder):
        if file.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(pose_image_folder, file)
            img = Image.open(img_path)
            width, height = img.size
            img = img.resize((400, int(400*height/width)))
            results = pose.process(np.array(img))
            # print(results)
            # print(img_path)
            if results:
                keypoint = helper.convert_mediapipe_to_openpose(
                results.pose_landmarks.landmark, height, width)
                pose_images.append([img,keypoint])

# Image cycling
image_sequence = []
best_score_image_sequence = []
current_image_index = -1
max_score = 0
start = False
running = True

def show_next_image():
    global current_image_index, max_score,start,running
    if current_image_index+1 < len(image_sequence):
        current_image_index+=1
        # print(current_image_index)
        img = ImageTk.PhotoImage(pose_images[image_sequence[current_image_index]][0])
        pose_image_label.config(image=img)
        pose_image_label.image = img
       
        max_score = 0
        root.after(5000, show_next_image)
    else:
        # print(best_score_image_sequence)
        start = False
        running = False
        show_results()
        # show_pose_sequence_gallery()

def start_game():
    global image_sequence, current_image_index,best_score_image_sequence,start
    count = image_count_var.get()
    if pose_images:
        image_sequence = random.sample(range(len(pose_images)), min(count, len(pose_images)))
        # print(image_sequence)
        # show_pose_sequence_gallery()

        best_score_image_sequence = [0]*len(image_sequence)
        current_image_index = -1
        start = True
        root.after(5000, show_next_image)
        # show_next_image()
    print("Game started!")
    
def reset_game():
    global current_image_index, max_score, start, running

    # Reset state
    current_image_index = -1
    max_score = 0
    start = False
    running = True

    # Clear right_frame content (pose results)
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Re-add video_label and restart the webcam loop
    video_label.pack(expand=True)
    update_frame()

    # Reset left panel pose image display
    pose_image_label.config(image="")
    pose_image_label.image = None


start_button = ttk.Button(left_frame, text="Start Game", command=start_game)
start_button.pack(pady=10)




# --- Right side: Webcam feed ---
right_frame = tk.Frame(root, bg="black")
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

video_label = tk.Label(right_frame)
video_label.pack(expand=True)


def show_results():
    # Clear webcam area
    video_label.pack_forget()

    result_title = tk.Label(right_frame, text="📊 Pose Comparison Results", font=("Helvetica", 16), bg="black", fg="white")
    result_title.pack(pady=10)

    # Scrollable area setup
    canvas = tk.Canvas(right_frame, bg="black", highlightthickness=0)
    scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="black")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Display side-by-side comparisons
    for score, user_img, _, idx in best_score_image_sequence:
        # Challenge pose
        orig_img = pose_images[image_sequence[idx]][0]
        orig_width, orig_height = orig_img.size
        orig_img = orig_img.resize((400, int(400 * orig_height / orig_width)))
        orig_tk = ImageTk.PhotoImage(orig_img)

        # User pose
        user_width, user_height = user_img.size
        user_img = user_img.resize((400, int(400 * user_height / user_width)))
        user_tk = ImageTk.PhotoImage(user_img)

        # Create row frame
        row = tk.Frame(scrollable_frame, bg="black")
        row.pack(pady=10)

        # Original pose image
        tk.Label(row, text=f"Pose #{idx + 1}", fg="white", bg="black", font=("Helvetica", 12)).pack()
        tk.Label(row, image=orig_tk, bg="black").pack(side="left", padx=10)
        tk.Label(row, image=user_tk, bg="black").pack(side="right", padx=10)

        # Keep image references
        row.image1 = orig_tk
        row.image2 = user_tk

        # Score
        tk.Label(scrollable_frame, text=f"Score: {score:.2f}", fg="lightgreen", bg="black", font=("Helvetica", 12)).pack()
        btn = tk.Button(
            scrollable_frame,
            text="👁 See Details",
            command=lambda idx=idx, user_img=user_img.copy(), score=score: show_detailed_comparison(idx, user_img, score),
            font=("Helvetica", 10),
            bg="white"
        )
        btn.pack(pady=5)

            # Try Again button
    try_again_btn = tk.Button(scrollable_frame, text="🔄 Try Again", font=("Helvetica", 12, "bold"),
                              command=reset_game, bg="white", fg="black")
    try_again_btn.pack(pady=20)

def show_detailed_comparison(idx, user_img, score):
    detail_window = tk.Toplevel(root)
    detail_window.title(f"Detail View – Pose #{idx + 1}")
    detail_window.configure(bg="white")
    detail_window.geometry("850x600")

    # Create scrollable canvas
    canvas = tk.Canvas(detail_window, bg="white", highlightthickness=0)
    scrollbar = tk.Scrollbar(detail_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg="white")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # --- Header ---
    tk.Label(scrollable_frame, text=f"Pose #{idx + 1} Comparison", font=("Helvetica", 14, "bold"), bg="white").pack(pady=10)
    tk.Label(scrollable_frame, text=f"Score: {score:.2f}", font=("Helvetica", 12), bg="white").pack(pady=5)

    # --- Pose Images Frame ---
    frame = tk.Frame(scrollable_frame, bg="white")
    frame.pack()

    # --- Original Pose Image ---
    orig_img = pose_images[image_sequence[idx]][0]
    orig_img = np.array(orig_img)
    orig_img = helper.resize_image(orig_img, 400)
    results = pose.process(orig_img)

    if results.pose_landmarks:
        keypoint1 = helper.convert_mediapipe_to_openpose(
            results.pose_landmarks.landmark, orig_img.shape[1], orig_img.shape[0])
        orig_img = helper.draw_bodypose(orig_img, keypoint1)

    orig_tk = ImageTk.PhotoImage(Image.fromarray(orig_img))

    # --- User Pose Image ---
    user_img = np.array(user_img)
    user_img = helper.resize_image(user_img, 400)
    results = pose.process(user_img)
    if results.pose_landmarks:
        keypoint2 = helper.convert_mediapipe_to_openpose(
            results.pose_landmarks.landmark, user_img.shape[1], user_img.shape[0])
        user_img = helper.draw_bodypose(user_img, keypoint2)

    user_tk = ImageTk.PhotoImage(Image.fromarray(user_img))

    # --- Comparison Row ---
    orig_label = tk.Label(frame, image=orig_tk, bg="white")
    user_label = tk.Label(frame, image=user_tk, bg="white")
    orig_label.grid(row=0, column=0, padx=10)
    user_label.grid(row=0, column=1, padx=10)

    # Keep references
    orig_label.image = orig_tk
    user_label.image = user_tk

    # --- GUIDE IMAGE ---
    canvas_img, score, guide = compare_keypoints.compare_keypoints(keypoint1, keypoint2, user_img, gpt=True)
    canvas_img = helper.resize_image(canvas_img, 400)
    guide_imgtk = ImageTk.PhotoImage(Image.fromarray(canvas_img))

    canvas_label = tk.Label(scrollable_frame, image=guide_imgtk, bg="white")
    canvas_label.image = guide_imgtk 
    canvas_label.pack(pady=10)

    # --- GUIDE TEXT ---
    guide_text = tk.Label(
        scrollable_frame,
        text=f"📝 Guide:\n{guide}",
        wraplength=700,
        justify="left",
        font=("Helvetica", 12),
        bg="white",
        fg="black",
        padx=10,
        pady=10
    )
    guide_text.pack(pady=5)



def update_frame():
    global max_score,start, running
    if not running:
        return
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = helper.resize_image(frame, 400)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        if start:
            results = pose.process(frame)
            # if results.pose_landmarks:
            #     mp_drawing.draw_landmarks(
            #         frame,
            #         results.pose_landmarks,
            #         mp_pose.POSE_CONNECTIONS,
            #         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            #         mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            #     )
            if results and results.pose_landmarks:
                keypoint = helper.convert_mediapipe_to_openpose(
                results.pose_landmarks.landmark, frame.shape[1], frame.shape[0])
                _,score = compare_keypoints.compare_keypoints(pose_images[image_sequence[current_image_index]][1],keypoint,guide=False)
                if score>max_score:
                    best_score_image_sequence[current_image_index] = [score,img,keypoint,current_image_index]
                    max_score = score
        
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
