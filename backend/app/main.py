from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import time


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SignFlow Backend",
    version="1.0.0"
)


# ============================================================
# CONNECTION MANAGER
# ============================================================

class ConnectionManager:

    def __init__(self):

        self.active_connections = []

        self.recognition_connection = None

        # Prevent sending camera frames too quickly.
        self.last_frame_time = 0.0

        self.frame_interval = 0.10  # ~10 FPS


    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    async def connect(self, websocket):

        await websocket.accept()

        if websocket not in self.active_connections:

            self.active_connections.append(
                websocket
            )

        print("WebSocket connected.")


    # --------------------------------------------------------
    # DISCONNECT
    # --------------------------------------------------------

    def disconnect(self, websocket):

        if websocket in self.active_connections:

            self.active_connections.remove(
                websocket
            )

        if self.recognition_connection is websocket:

            self.recognition_connection = None

            print(
                "Recognition client disconnected."
            )

        print(
            "WebSocket disconnected."
        )


    # --------------------------------------------------------
    # REGISTER RECOGNITION
    # --------------------------------------------------------

    def register_recognition(
        self,
        websocket
    ):

        # If another recognition connection exists,
        # remove it from the active list.

        if (
            self.recognition_connection is not None
            and
            self.recognition_connection is not websocket
        ):

            old = self.recognition_connection

            if old in self.active_connections:

                self.active_connections.remove(
                    old
                )


        self.recognition_connection = websocket

        print(
            "Recognition client registered."
        )


    # --------------------------------------------------------
    # SEND JSON
    # --------------------------------------------------------

    async def safe_send_json(
        self,
        websocket,
        data
    ):

        try:

            await websocket.send_json(
                data
            )

            return True

        except Exception:

            # Do NOT print repeatedly.
            # Just remove the dead connection.

            self.disconnect(
                websocket
            )

            return False


    # --------------------------------------------------------
    # SEND BYTES
    # --------------------------------------------------------

    async def safe_send_bytes(
        self,
        websocket,
        data
    ):

        try:

            await websocket.send_bytes(
                data
            )

            return True

        except Exception:

            # Remove dead frontend connection.
            self.disconnect(
                websocket
            )

            return False


    # --------------------------------------------------------
    # BROADCAST JSON
    # --------------------------------------------------------

    async def broadcast_json(
        self,
        data
    ):

        connections = list(
            self.active_connections
        )

        for websocket in connections:

            # Never send recognition messages
            # back to recognition.

            if websocket is self.recognition_connection:

                continue

            await self.safe_send_json(
                websocket,
                data
            )


    # --------------------------------------------------------
    # BROADCAST CAMERA FRAME
    # --------------------------------------------------------

    async def broadcast_frame(
        self,
        frame
    ):

        current_time = time.monotonic()

        # Limit camera forwarding to ~10 FPS.

        if (
            current_time
            - self.last_frame_time
            < self.frame_interval
        ):

            return

        self.last_frame_time = current_time


        connections = list(
            self.active_connections
        )


        for websocket in connections:

            if websocket is self.recognition_connection:

                continue

            await self.safe_send_bytes(
                websocket,
                frame
            )


    # --------------------------------------------------------
    # SEND CONTROL TO RECOGNITION
    # --------------------------------------------------------

    async def send_control(
        self,
        action
    ):

        websocket = (
            self.recognition_connection
        )


        if websocket is None:

            print(
                "Recognition client is not connected."
            )

            return False


        try:

            await websocket.send(
                json.dumps({

                    "type": "control",

                    "action": action

                })
            )

            print(
                "Control sent to recognition:",
                action
            )

            return True


        except Exception:

            self.disconnect(
                websocket
            )

            return False


manager = ConnectionManager()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {

        "name": "SignFlow Backend",

        "status": "running",

        "websocket": "/ws"

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {

        "status": "ok"

    }


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await manager.connect(
        websocket
    )


    client_role = "frontend"


    try:

        # ----------------------------------------------------
        # CONNECTION MESSAGE
        # ----------------------------------------------------

        await websocket.send_json({

            "type": "connection",

            "status": "connected",

            "message": (
                "Connected to SignFlow backend"
            )

        })


        # ----------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------

        while True:

            message = await websocket.receive()


            # =================================================
            # DISCONNECT
            # =================================================

            if (
                message.get("type")
                == "websocket.disconnect"
            ):

                break


            # =================================================
            # BINARY CAMERA FRAME
            # =================================================

            if message.get("bytes") is not None:

                frame = message["bytes"]


                # Only recognition is allowed
                # to send camera frames.

                if (
                    websocket
                    is manager.recognition_connection
                ):

                    await manager.broadcast_frame(
                        frame
                    )


                continue


            # =================================================
            # TEXT
            # =================================================

            text = message.get(
                "text"
            )


            if text is None:

                continue


            try:

                data = json.loads(
                    text
                )

            except json.JSONDecodeError:

                await manager.safe_send_json(

                    websocket,

                    {

                        "type": "error",

                        "message": (
                            "Invalid JSON message"
                        )

                    }

                )

                continue


            message_type = data.get(
                "type"
            )


            # =================================================
            # REGISTER
            # =================================================

            if message_type == "register":

                role = data.get(
                    "role",
                    "frontend"
                )

                client_role = role


                print(
                    f"Client registered as: {role}"
                )


                if role == "recognition":

                    manager.register_recognition(
                        websocket
                    )


                    await manager.safe_send_json(

                        websocket,

                        {

                            "type": "registration",

                            "role": "recognition",

                            "status": "registered"

                        }

                    )

                else:

                    await manager.safe_send_json(

                        websocket,

                        {

                            "type": "registration",

                            "role": role,

                            "status": "registered"

                        }

                    )

                continue


            # =================================================
            # PING
            # =================================================

            if message_type == "ping":

                await manager.safe_send_json(

                    websocket,

                    {

                        "type": "pong",

                        "message": (
                            "SignFlow backend is working"
                        )

                    }

                )

                continue


            # =================================================
            # FRONTEND CONTROL
            # =================================================

            if message_type == "control":

                action = data.get(
                    "action"
                )


                allowed_actions = {

                    "enter",

                    "backspace",

                    "clear",

                    "quit"

                }


                if action not in allowed_actions:

                    await manager.safe_send_json(

                        websocket,

                        {

                            "type": "error",

                            "message": (
                                "Unknown control action"
                            )

                        }

                    )

                    continue


                print(
                    "Frontend control:",
                    action
                )


                success = (
                    await manager.send_control(
                        action
                    )
                )


                if not success:

                    await manager.safe_send_json(

                        websocket,

                        {

                            "type": "error",

                            "message": (
                                "Recognition client "
                                "is not connected"
                            )

                        }

                    )

                continue


            # =================================================
            # SIGN
            # =================================================

            if message_type == "sign":

                await manager.broadcast_json({

                    "type": "recognition",

                    "sign": data.get(
                        "sign",
                        ""
                    ),

                    "sentence": data.get(
                        "sentence",
                        ""
                    ),

                    "status": "recognizing"

                })

                continue


            # =================================================
            # LIVE RECOGNITION
            # =================================================

            if message_type == "recognition_live":

                await manager.broadcast_json({

                    "type": "recognition_live",

                    "sign": data.get(
                        "sign",
                        ""
                    ),

                    "distance": data.get(
                        "distance",
                        0
                    ),

                    "sentence": data.get(
                        "sentence",
                        ""
                    )

                })

                continue


            # =================================================
            # SENTENCE STATE
            # =================================================

            if message_type == "sentence_state":

                await manager.broadcast_json({

                    "type": "sentence_state",

                    "sentence": data.get(
                        "sentence",
                        ""
                    )

                })

                continue


            # =================================================
            # FINAL SENTENCE
            # =================================================

            if message_type in (
                "sentence_result",
                "sentence"
            ):

                sentence = data.get(
                    "sentence",
                    ""
                )

                translation = data.get(
                    "translation",
                    ""
                )


                print()
                print(
                    "=" * 50
                )

                print(
                    "FINAL SENTENCE"
                )

                print(
                    "English:",
                    sentence
                )

                print(
                    "Translation:",
                    translation
                )

                print(
                    "=" * 50
                )


                await manager.broadcast_json({

                    "type": "sentence_result",

                    "sentence": sentence,

                    "translation": translation

                })

                continue


            # =================================================
            # ERROR
            # =================================================

            if message_type == "error":

                await manager.broadcast_json({

                    "type": "error",

                    "message": data.get(
                        "message",
                        "Unknown error"
                    )

                })

                continue


            # =================================================
            # UNKNOWN MESSAGE
            # =================================================

            await manager.safe_send_json(

                websocket,

                {

                    "type": "error",

                    "message": (
                        "Unknown message type: "
                        f"{message_type}"
                    )

                }

            )


    except WebSocketDisconnect:

        print(
            f"{client_role} WebSocket disconnected."
        )


    except Exception as e:

        print(
            f"{client_role} WebSocket error:",
            e
        )


    finally:

        manager.disconnect(
            websocket
        )