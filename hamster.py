# hamster.py
# Looks at your face through the webcam, guesses your expression
# and shows a matching hamster picture in the top right corner.
#
# How to run:  press F5 in VS Code
# How to quit: click the camera window and press q (or Esc, or the X button)

import subprocess
import time
from collections import Counter, deque
from pathlib import Path

import cv2                # OpenCV: camera, drawing, windows
import mediapipe as mp    # Google's face tracking
import numpy as np        # math with lists of numbers (OpenCV needs it for polygons)
from mediapipe.tasks.python import BaseOptions, vision


# ---------------- settings ----------------

# Path(__file__) is the folder this script is in, so the files are found
# no matter where you run the program from.
MODEL_PATH = Path(__file__).with_name("face_landmarker.task")  # the AI model file
PHOTOS_DIR = Path(__file__).with_name("hamsterphotos")         # folder with the hamster pics
SOUNDS_DIR = Path(__file__).with_name("sounds")                # folder with the sounds
PHOTO_HEIGHT = 160   # how tall the hamster picture is, in pixels
CAMERA_INDEX = 0     # 0 = first camera. try 1 if you have more than one

# Colors for each expression.
# Careful: OpenCV uses BGR (blue, green, red), not RGB!
COLORS = {
    "Laughing": (0, 215, 255),    # gold
    "Happy": (0, 200, 0),         # green
    "Surprised": (255, 200, 0),   # light blue
    "Disgusted": (0, 140, 70),    # dark green
    "Angry": (0, 0, 255),         # red
    "Sad": (200, 80, 0),          # blue
    "Kissy": (180, 100, 255),     # pink
    "Sleepy": (150, 150, 150),    # gray
    "Neutral": (255, 255, 255),   # white
    "No face": (100, 100, 100),   # dark gray
}


# ---------------- guessing the expression ----------------

def classify(bs):
    # MediaPipe gives us "blendshapes": 52 numbers between 0 and 1 that say
    # how much each part of the face is moving. For example
    # bs["jawOpen"] = 0.0 means mouth closed, 1.0 means wide open.
    # Most of them come in pairs (left + right), so I average the two sides.
    smile = (bs["mouthSmileLeft"] + bs["mouthSmileRight"]) / 2
    frown = (bs["mouthFrownLeft"] + bs["mouthFrownRight"]) / 2
    brow_up = (bs["browInnerUp"] + bs["browOuterUpLeft"] + bs["browOuterUpRight"]) / 3
    brow_down = (bs["browDownLeft"] + bs["browDownRight"]) / 2
    jaw_open = bs["jawOpen"]
    eyes_wide = (bs["eyeWideLeft"] + bs["eyeWideRight"]) / 2
    eyes_closed = (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2
    sneer = (bs["noseSneerLeft"] + bs["noseSneerRight"]) / 2   # wrinkled nose
    pucker = bs["mouthPucker"]                                 # kissy lips

    # Checked from top to bottom, the first one that matches wins.
    # The order matters: a big smile with an open mouth is "Laughing",
    # so that has to be checked before "Happy".
    # If an expression triggers too easily or not at all, change the numbers here.
    if smile > 0.5 and jaw_open > 0.3:
        return "Laughing"
    if smile > 0.4:
        return "Happy"
    if jaw_open > 0.4 and (brow_up > 0.35 or eyes_wide > 0.3):
        return "Surprised"
    if sneer > 0.3:
        return "Disgusted"
    if brow_down > 0.4:
        return "Angry"
    if frown > 0.3 or (bs["browInnerUp"] > 0.4 and smile < 0.1):
        return "Sad"
    if pucker > 0.6:
        return "Kissy"
    if eyes_closed > 0.6:
        return "Sleepy"
    return "Neutral"   # nothing matched


# ---------------- the triangle mesh on the face ----------------

# MediaPipe finds 478 points on the face, each with its own number.
# Drawing all of them is way too busy, so I picked some important ones.
# (map of all the numbers: search "mediapipe face mesh landmarks" on google)
MESH_POINTS = [
    10, 297, 284, 389, 454, 361, 397, 379, 400, 152,   # edge of the face, right side + chin
    176, 150, 172, 132, 234, 162, 54, 67,              # edge of the face, left side
    70, 105, 107, 336, 334, 300,                       # eyebrows
    33, 133, 159, 145, 362, 263, 386, 374,             # eyes
    168, 6, 1, 98, 327, 2,                             # nose
    61, 291, 0, 17,                                    # mouth
    50, 280, 205, 425,                                 # cheeks
]

# Which points connect into triangles. Starts empty (None) and gets
# filled the first time we see a face, then reused every frame.
# If we worked it out again every frame the triangles would keep
# flipping around and the mesh would flicker.
_triangles = None


def build_triangles(pts):
    # Connects the points into triangles using "Delaunay triangulation".
    # It's a classic algorithm that makes nice fat triangles instead of
    # long thin ones. OpenCV has it built in (Subdiv2D).

    # Subdiv2D needs a rectangle that contains all the points,
    # so take the smallest/biggest x and y, plus a little extra room.
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    left, top = min(xs) - 10, min(ys) - 10
    width = max(xs) - min(xs) + 20
    height = max(ys) - min(ys) + 20

    subdiv = cv2.Subdiv2D((left, top, width, height))
    for p in pts:
        subdiv.insert((float(p[0]), float(p[1])))

    # OpenCV gives back triangles as corner coordinates (x1, y1, x2, y2, x3, y3),
    # but I want to know WHICH points they are, so the triangles can
    # move with the face later. This finds the point closest to (x, y).
    def nearest(x, y):
        best = 0
        for i in range(len(pts)):
            if (pts[i][0] - x) ** 2 + (pts[i][1] - y) ** 2 < (pts[best][0] - x) ** 2 + (pts[best][1] - y) ** 2:
                best = i
        return best

    triangles = set()   # a set skips duplicates automatically
    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():
        a, b, c = nearest(x1, y1), nearest(x2, y2), nearest(x3, y3)
        if a != b and b != c and a != c:   # skip broken triangles
            triangles.add(tuple(sorted((a, b, c))))
    return list(triangles)


def draw_face_lines(frame, landmarks, color):
    global _triangles   # "global" lets this function change the variable outside it

    # The landmarks come as numbers from 0 to 1 (0.5 = middle of the image),
    # so multiply by the image size to get real pixel positions.
    h, w = frame.shape[:2]
    pts = []
    for i in MESH_POINTS:
        pts.append((int(landmarks[i].x * w), int(landmarks[i].y * h)))

    if _triangles is None:
        _triangles = build_triangles(pts)

    # Trick for see-through lines: draw on a copy of the image,
    # then mix the copy with the original (60% copy, 40% original).
    overlay = frame.copy()
    for a, b, c in _triangles:
        triangle = np.array([pts[a], pts[b], pts[c]])
        cv2.polylines(overlay, [triangle], True, color, 1, cv2.LINE_AA)   # LINE_AA = smooth edges
    for p in pts:
        cv2.circle(overlay, p, 2, color, -1, cv2.LINE_AA)   # -1 = filled circle
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Now the square around the face. Use ALL 478 points to find
    # how far the face goes left, right, up and down.
    all_x = [int(p.x * w) for p in landmarks]
    all_y = [int(p.y * h) for p in landmarks]
    center_x = (min(all_x) + max(all_x)) // 2
    center_y = (min(all_y) + max(all_y)) // 2
    face_size = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    half = int(face_size * 0.6)   # 0.6 instead of 0.5 = a bit of space around the face

    x1, y1 = center_x - half, center_y - half   # top left corner
    x2, y2 = center_x + half, center_y + half   # bottom right corner
    arm = half // 4                             # how long each corner line is

    # Only draw the corners (like a camera viewfinder), not the whole square.
    # Each corner is two short lines: one going sideways, one going up/down.
    # dx and dy say which direction the lines go from that corner (1 or -1).
    corners = [
        ((x1, y1), (1, 1)),     # top left: lines go right and down
        ((x2, y1), (-1, 1)),    # top right: left and down
        ((x1, y2), (1, -1)),    # bottom left: right and up
        ((x2, y2), (-1, -1)),   # bottom right: left and up
    ]
    for (px, py), (dx, dy) in corners:
        cv2.line(frame, (px, py), (px + dx * arm, py), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * arm), color, 2, cv2.LINE_AA)


# ---------------- the corner: hamster picture or text ----------------

def load_photos():
    # Looks in hamsterphotos/ for a picture for each expression.
    # The file name must match the expression, e.g. happy.gif or sad.png.
    photos = {}
    for label in COLORS:
        for ext in (".gif", ".png", ".jpg", ".jpeg"):
            path = PHOTOS_DIR / f"{label.lower()}{ext}"
            if not path.exists():
                continue   # not this one, try the next file type

            # cv2.imread can't open GIFs, but VideoCapture can (it treats it like a tiny video)
            ok, img = cv2.VideoCapture(str(path)).read()
            if ok:
                # resize to PHOTO_HEIGHT tall, keeping the shape (so it doesn't look squished)
                scale = PHOTO_HEIGHT / img.shape[0]
                new_width = int(img.shape[1] * scale)
                photos[label] = cv2.resize(img, (new_width, PHOTO_HEIGHT), interpolation=cv2.INTER_AREA)
                break   # found it, no need to check the other file types
    return photos


def draw_photo(frame, photo, color):
    h, w = photo.shape[:2]
    x = frame.shape[1] - w - 14   # 14 pixels from the right edge
    y = 14                        # 14 pixels from the top
    border = 4

    # First a filled rectangle a bit bigger than the photo (that's the colored frame)...
    cv2.rectangle(frame, (x - border, y - border), (x + w + border - 1, y + h + border - 1), color, -1)
    # ...then paste the photo on top. An image is just a big grid of pixels,
    # so we can copy the photo into a piece of that grid.
    frame[y:y + h, x:x + w] = photo


def draw_label(frame, text):
    # Used when there's no picture (like "No face"): writes the word on a black box.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, pad = 1.2, 3, 12

    # Measure the text first so the box is exactly the right size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = frame.shape[1] - text_w - 2 * pad - 10
    y = 10

    cv2.rectangle(frame, (x, y), (x + text_w + 2 * pad, y + text_h + baseline + 2 * pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + pad, y + pad + text_h), font, scale,
                COLORS.get(text, (255, 255, 255)), thickness, cv2.LINE_AA)


# ---------------- sounds ----------------

def load_sounds():
    # Same idea as load_photos: sounds/happy.mp3 goes with "Happy" and so on.
    # We only remember where the files are, the player opens them itself.
    sounds = {}
    for label in COLORS:
        for ext in (".mp3", ".wav", ".ogg", ".flac"):
            path = SOUNDS_DIR / f"{label.lower()}{ext}"
            if path.exists():
                sounds[label] = path
                break
    return sounds


def play_sound(path, old_player):
    # Stop the sound that's still playing (if there is one),
    # so two sounds never play on top of each other.
    if old_player is not None and old_player.poll() is None:   # poll() is None = still running
        old_player.terminate()
    # pw-play is PipeWire's sound player (comes with Linux).
    # Popen starts it in the background, so the camera doesn't freeze while it plays.
    return subprocess.Popen(["pw-play", str(path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------- main program ----------------

def main():
    global _triangles

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    photos = load_photos()
    sounds = load_sounds()

    # Set up the face tracker
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,   # VIDEO mode remembers the face between frames = smoother
        num_faces=1,                             # only track one person
        # How sure it must be that it's a real face (0 to 1).
        # At 0.5 it thought my Misfits poster was a face lol, 0.7 fixed it.
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        output_face_blendshapes=True,            # we need these for classify()
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        # Only one program can use the camera at a time
        raise SystemExit("Could not open the camera - is another copy of hamster.py "
                         "(or a stuck debug session) still running? Stop it and try again.")

    # Remembers the last 7 guesses. We show the one that came up most,
    # otherwise the picture flickers like crazy between two expressions.
    # (deque with maxlen = a list that throws out the oldest item when it's full)
    history = deque(maxlen=7)
    start = time.monotonic()   # VIDEO mode needs to know the time of each frame
    last_label = None          # the expression from the frame before, to notice when it changes
    player = None              # the sound that's playing right now (None = nothing)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:   # runs once per camera frame, until we break out
            ok, frame = cap.read()
            if not ok:
                break   # camera stopped working

            frame = cv2.flip(frame, 1)   # mirror it, like looking in a mirror

            # OpenCV images are BGR but MediaPipe wants RGB, so convert
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(image, timestamp_ms)

            if result.face_blendshapes:   # found a face
                # turn the list of blendshapes into a dict, e.g. {"jawOpen": 0.12, ...}
                scores = {}
                for b in result.face_blendshapes[0]:
                    scores[b.category_name] = b.score

                history.append(classify(scores))
                label = Counter(history).most_common(1)[0][0]   # most common recent guess
                draw_face_lines(frame, result.face_landmarks[0], COLORS[label])
            else:   # no face
                _triangles = None   # make a fresh mesh next time a face shows up
                history.clear()
                label = "No face"

            # Play the sound once, only at the moment the face changes
            if label != last_label and label in sounds:
                player = play_sound(sounds[label], player)
            last_label = label

            if label in photos:
                draw_photo(frame, photos[label], COLORS[label])
            else:
                draw_label(frame, label)   # no picture for this one, show the word

            cv2.imshow("Hamster Face", frame)

            # waitKey(1) waits 1 ms for a key press. 27 is the Esc key.
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            # stop if the window was closed with the X button
            if cv2.getWindowProperty("Hamster Face", cv2.WND_PROP_VISIBLE) < 1:
                break

    # clean up so the camera is free for other programs
    if player is not None:
        player.terminate()   # don't keep playing after the window is closed
    cap.release()
    cv2.destroyAllWindows()


# This only runs main() when you start this file directly,
# not when another file does "import hamster".
if __name__ == "__main__":
    main()
