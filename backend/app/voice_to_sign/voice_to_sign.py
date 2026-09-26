"""
SIgnFlow
Continuous Voice -> Text -> ISL Signs -> Video Output
"""

from .voice_input import listen_once
from .text_processor import process_text
from .sign_mapper import map_text_to_signs
from .sign_player import play_sequence


def main():

    print()
    print("=" * 60)
    print("                 SIgnFlow")
    print("          CONTINUOUS VOICE -> ISL")
    print("=" * 60)

    print()
    print("Speak a sentence.")
    print("After the sign videos finish, SIgnFlow will listen again.")
    print("Press Ctrl+C in PowerShell to stop.")
    print()

    try:

        while True:

            # ---------------------------------------------
            # 1. LISTEN
            # ---------------------------------------------

            print()
            print("-" * 60)
            print("[LISTENING]")
            print("Speak now...")
            print("-" * 60)

            text = listen_once()

            if not text:
                print()
                print("[SYSTEM] Nothing recognized.")
                print("[SYSTEM] Returning to listening...")
                continue

            # ---------------------------------------------
            # 2. PROCESS TEXT
            # ---------------------------------------------

            print()
            print("[PROCESSING]")

            cleaned_text = process_text(text)

            if not cleaned_text:
                print("[SYSTEM] Empty text.")
                continue

            # ---------------------------------------------
            # 3. MAP TO SIGNS
            # ---------------------------------------------

            print()
            print("[SIGN MAPPING]")

            signs = map_text_to_signs(cleaned_text)

            if not signs:

                print()
                print("[SYSTEM] No supported signs found.")
                print("[TEXT]", cleaned_text)
                print("[SYSTEM] Returning to listening...")

                continue

            # ---------------------------------------------
            # 4. SHOW SEQUENCE
            # ---------------------------------------------

            print()
            print("=" * 60)
            print("              ISL SIGN SEQUENCE")
            print("=" * 60)

            print(
                " -> ".join(
                    sign.upper()
                    for sign in signs
                )
            )

            print("=" * 60)

            # ---------------------------------------------
            # 5. PLAY VIDEOS
            # ---------------------------------------------

            print()
            print("[OUTPUT] Playing ISL videos...")

            play_sequence(signs)

            print()
            print("[SYSTEM] Finished.")
            print("[SYSTEM] Returning to listening...")

    except KeyboardInterrupt:

        print()
        print()
        print("=" * 60)
        print("             SIgnFlow STOPPED")
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()