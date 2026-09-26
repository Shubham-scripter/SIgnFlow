"""
SIgnFlow - ISL Video Sign Player

Temporary prototype assets are being used.
The files can later be replaced with actual ISL videos
without changing the Python code.
"""

from pathlib import Path
import tkinter as tk
import cv2
from PIL import Image, ImageTk


BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets" / "isl"


SIGN_FILES = {
    "you": "you.mp4",
    "name": "name.mp4",
    "what": "what.mp4",
}


class SignPlayer:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title("SIgnFlow - ISL Output")
        self.root.geometry("900x650")
        self.root.configure(bg="#111111")

        self.title_label = tk.Label(
            self.root,
            text="SIgnFlow - ISL OUTPUT",
            font=("Segoe UI", 24, "bold"),
            fg="white",
            bg="#111111",
        )
        self.title_label.pack(pady=15)

        self.status_label = tk.Label(
            self.root,
            text="Waiting...",
            font=("Segoe UI", 16),
            fg="#bbbbbb",
            bg="#111111",
        )
        self.status_label.pack(pady=5)

        self.video_label = tk.Label(
            self.root,
            bg="#111111",
        )
        self.video_label.pack(
            expand=True,
            fill="both",
            padx=20,
            pady=10,
        )

        self.sequence_label = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 14),
            fg="#888888",
            bg="#111111",
        )
        self.sequence_label.pack(pady=15)

        self.cap = None
        self.current_sign = None
        self.signs = []
        self.index = 0

    def get_video_path(self, sign):

        sign = str(sign).strip().lower()

        if sign not in SIGN_FILES:
            print(f"[SIGN PLAYER] Unsupported sign: {sign}")
            return None

        path = ASSET_DIR / SIGN_FILES[sign]

        if not path.exists():
            print("[SIGN PLAYER] Missing video:")
            print(path)
            return None

        return path

    def play_sequence(self, signs):

        if not signs:
            print("[SIGN PLAYER] Empty sequence.")
            return

        self.signs = signs
        self.index = 0

        self.sequence_label.config(
            text="  →  ".join(
                str(sign).upper()
                for sign in signs
            )
        )

        self.play_current_sign()

    def play_current_sign(self):

        if self.index >= len(self.signs):

            self.status_label.config(
                text="Sequence finished"
            )

            self.video_label.config(
                image=""
            )

            return

        sign = self.signs[self.index]

        self.current_sign = sign

        self.status_label.config(
            text=f"ISL Sign {self.index + 1} / {len(self.signs)}   •   {sign.upper()}"
        )

        video_path = self.get_video_path(sign)

        if video_path is None:

            self.index += 1

            self.root.after(
                500,
                self.play_current_sign
            )

            return

        print()
        print(f"[SIGN PLAYER] Playing: {sign}")
        print(f"[VIDEO] {video_path}")

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(
            str(video_path)
        )

        if not self.cap.isOpened():

            print(
                f"[SIGN PLAYER ERROR] Could not open: {video_path}"
            )

            self.index += 1

            self.root.after(
                500,
                self.play_current_sign
            )

            return

        self.play_video_frame()

    def play_video_frame(self):

        if self.cap is None:
            return

        success, frame = self.cap.read()

        if not success:

            self.cap.release()
            self.cap = None

            self.index += 1

            self.root.after(
                500,
                self.play_current_sign
            )

            return

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Fit video inside the window.

        height, width, _ = frame.shape

        max_width = 820
        max_height = 480

        scale = min(
            max_width / width,
            max_height / height,
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        frame = cv2.resize(
            frame,
            (new_width, new_height),
        )

        image = Image.fromarray(frame)

        photo = ImageTk.PhotoImage(
            image=image
        )

        self.video_label.config(
            image=photo
        )

        self.video_label.image = photo

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

        delay = int(
            1000 / fps
        )

        self.root.after(
            delay,
            self.play_video_frame
        )

    def run(self):

        self.root.mainloop()


def play_sequence(signs):

    player = SignPlayer()

    player.root.after(
        300,
        lambda: player.play_sequence(
            signs
        )
    )

    player.run()


def check_assets():

    print()
    print("=" * 50)
    print("             ISL VIDEO CHECK")
    print("=" * 50)

    for sign, filename in SIGN_FILES.items():

        path = ASSET_DIR / filename

        if path.exists():

            print(
                f"[OK]      {sign:<10} {filename}"
            )

        else:

            print(
                f"[MISSING] {sign:<10} {filename}"
            )

    print("=" * 50)


if __name__ == "__main__":

    check_assets()

    print()
    print("Testing:")
    print("YOU -> NAME -> WHAT")

    play_sequence(
        [
            "you",
            "name",
            "what",
        ]
    )