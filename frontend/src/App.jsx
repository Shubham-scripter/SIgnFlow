import { useEffect, useRef, useState } from "react";
import "./App.css";

const WS_URL = "ws://127.0.0.1:8000/ws";

function App() {
  const socketRef = useRef(null);
  const frameUrlRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [cameraFrame, setCameraFrame] = useState(null);

  const [recognizedSign, setRecognizedSign] = useState("Waiting...");
  const [sentence, setSentence] = useState("");
  const [translation, setTranslation] = useState("");

  const [status, setStatus] = useState("Connecting...");
  const [targetLanguage, setTargetLanguage] = useState("Hindi");

  // ============================================================
  // SEND CONTROL COMMAND
  // ============================================================

  const sendControl = (action) => {
    const ws = socketRef.current;

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }

    ws.send(
      JSON.stringify({
        type: "control",
        action,
      })
    );
  };

  // ============================================================
  // WEBSOCKET
  // ============================================================

  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.binaryType = "arraybuffer";

    socketRef.current = ws;

    // ----------------------------------------------------------
    // CONNECTED
    // ----------------------------------------------------------

    ws.onopen = () => {
      setConnected(true);
      setStatus("AI camera connected");

      ws.send(
        JSON.stringify({
          type: "register",
          role: "frontend",
        })
      );
    };

    // ----------------------------------------------------------
    // RECEIVE DATA
    // ----------------------------------------------------------

    ws.onmessage = (event) => {
      // ========================================================
      // CAMERA FRAME
      // ========================================================

      if (event.data instanceof ArrayBuffer) {
        const blob = new Blob([event.data], {
          type: "image/jpeg",
        });

        const newUrl = URL.createObjectURL(blob);

        const oldUrl = frameUrlRef.current;

        frameUrlRef.current = newUrl;

        setCameraFrame(newUrl);

        if (oldUrl) {
          URL.revokeObjectURL(oldUrl);
        }

        return;
      }

      // ========================================================
      // JSON MESSAGE
      // ========================================================

      try {
        const data = JSON.parse(event.data);

        // ------------------------------------------------------
        // CONNECTION
        // ------------------------------------------------------

        if (data.type === "connection") {
          setStatus("AI camera connected");
        }

        // ------------------------------------------------------
        // RECOGNITION
        // ------------------------------------------------------

        if (data.type === "recognition") {
          if (data.sign) {
            setRecognizedSign(
              data.sign.toUpperCase()
            );
          }

          setSentence(
            data.sentence ?? ""
          );

          setStatus("Recognizing");
        }

        // ------------------------------------------------------
        // LIVE RECOGNITION
        // ------------------------------------------------------

        if (data.type === "recognition_live") {
          if (data.sign) {
            setRecognizedSign(
              data.sign.toUpperCase()
            );
          }

          setSentence(
            data.sentence ?? ""
          );

          setStatus("Recognizing");
        }

        // ------------------------------------------------------
        // SENTENCE STATE
        // ------------------------------------------------------

        if (data.type === "sentence_state") {
          setSentence(
            data.sentence ?? ""
          );
        }

        // ------------------------------------------------------
        // FINAL SENTENCE
        // ------------------------------------------------------

        if (data.type === "sentence_result") {
          setSentence(
            data.sentence ?? ""
          );

          setTranslation(
            data.translation ?? ""
          );

          setStatus("Translation ready");
        }

        // ------------------------------------------------------
        // ERROR
        // ------------------------------------------------------

        if (data.type === "error") {
          setStatus(
            data.message || "Backend error"
          );

          console.error(
            data.details || data.message
          );
        }
      } catch (error) {
        console.error(
          "Invalid WebSocket message:",
          error
        );
      }
    };

    // ----------------------------------------------------------
    // DISCONNECTED
    // ----------------------------------------------------------

    ws.onclose = () => {
      setConnected(false);
      setStatus("Backend disconnected");
    };

    // ----------------------------------------------------------
    // ERROR
    // ----------------------------------------------------------

    ws.onerror = () => {
      setConnected(false);
      setStatus("Connection error");
    };

    // ----------------------------------------------------------
    // CLEANUP
    // ----------------------------------------------------------

    return () => {
      ws.close();

      if (frameUrlRef.current) {
        URL.revokeObjectURL(
          frameUrlRef.current
        );

        frameUrlRef.current = null;
      }
    };
  }, []);

  // ============================================================
  // KEYBOARD CONTROLS
  // ============================================================

  useEffect(() => {
    const handleKeyDown = (event) => {
      const tag = event.target?.tagName;

      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT"
      ) {
        return;
      }

      if (event.key === "Enter") {
        event.preventDefault();

        sendControl("enter");
      }

      else if (event.key === "Backspace") {
        event.preventDefault();

        sendControl("backspace");
      }

      else if (
        event.key.toLowerCase() === "c"
      ) {
        event.preventDefault();

        sendControl("clear");
      }
    };

    window.addEventListener(
      "keydown",
      handleKeyDown
    );

    return () => {
      window.removeEventListener(
        "keydown",
        handleKeyDown
      );
    };
  }, []);

  // ============================================================
  // CLEAR LOCAL UI
  // ============================================================

  const clearInterface = () => {
    sendControl("clear");

    setSentence("");
    setTranslation("");
    setRecognizedSign("Waiting...");
    setStatus(
      connected
        ? "Ready"
        : "Backend disconnected"
    );
  };

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            SF
          </div>

          <div className="brand-text">

            <h1>SignFlow</h1>

            <p>
              Real-time sign language communication
            </p>

          </div>

        </div>

        <div className="topbar-right">

          <div className="language-selector">

            <span>
              Language
            </span>

            <select
              value={targetLanguage}
              onChange={(event) =>
                setTargetLanguage(
                  event.target.value
                )
              }
            >
              <option value="Hindi">
                Hindi
              </option>

              <option value="English">
                English
              </option>

              <option value="Spanish">
                Spanish
              </option>

              <option value="French">
                French
              </option>
            </select>

          </div>

          <div
            className={
              connected
                ? "system-status online"
                : "system-status offline"
            }
          >

            <span className="status-dot" />

            <span>
              {connected
                ? "System Ready"
                : "Disconnected"}
            </span>

          </div>

        </div>

      </header>


      {/* ======================================================
          MAIN CONTENT
      ====================================================== */}

      <main className="dashboard">

        {/* ====================================================
            CAMERA
        ==================================================== */}

        <section className="camera-panel">

          <div className="panel-top">

            <div>

              <div className="eyebrow">
                LIVE CAMERA
              </div>

              <h2>
                Sign recognition
              </h2>

            </div>

            <div className="live-pill">

              <span />

              LIVE

            </div>

          </div>


          <div className="camera-wrapper">

            {cameraFrame ? (

              <img
                src={cameraFrame}
                className="camera-feed"
                alt="SignFlow camera feed"
              />

            ) : (

              <div className="camera-empty">

                <div className="camera-empty-icon">
                  <span />
                  <span />
                  <span />
                </div>

                <h3>
                  Waiting for camera
                </h3>

                <p>
                  Start the SignFlow recognition
                  engine to begin.
                </p>

              </div>

            )}

            {/* Camera overlay */}

            <div className="camera-overlay">

              <div className="corner top-left" />
              <div className="corner top-right" />
              <div className="corner bottom-left" />
              <div className="corner bottom-right" />

            </div>

            <div className="camera-label">
              AI VISION
            </div>

          </div>


          <div className="camera-footer">

            <div className="camera-status">

              <span
                className={
                  connected
                    ? "status-dot"
                    : "status-dot offline-dot"
                }
              />

              <span>
                {status}
              </span>

            </div>

            <div className="camera-hint">
              Position your hands inside the frame
            </div>

          </div>

        </section>


        {/* ====================================================
            RECOGNITION PANEL
        ==================================================== */}

        <section className="recognition-panel">

          <div className="panel-top">

            <div>

              <div className="eyebrow">
                RECOGNITION
              </div>

              <h2>
                Live interpretation
              </h2>

            </div>

            <div className="ai-indicator">
              AI
            </div>

          </div>


          {/* CURRENT SIGN */}

          <div className="sign-display">

            <div className="display-label">
              CURRENT SIGN
            </div>

            <div className="sign-word">
              {recognizedSign}
            </div>

            <div className="sign-line" />

            <p>
              Hand gesture detected by
              SignFlow vision engine
            </p>

          </div>


          {/* SENTENCE */}

          <div className="sentence-display">

            <div className="display-label">
              SENTENCE
            </div>

            <div
              className={
                sentence
                  ? "sentence-value"
                  : "sentence-value empty"
              }
            >
              {sentence ||
                "Your recognized signs will appear here..."}
            </div>

          </div>


          {/* STATUS */}

          <div className="recognition-status">

            <div className="recognition-status-icon">
              ✓
            </div>

            <div>

              <strong>
                {connected
                  ? "Recognition active"
                  : "Recognition unavailable"}
              </strong>

              <span>
                {connected
                  ? "SignFlow is listening for gestures"
                  : "Start the backend to reconnect"}
              </span>

            </div>

          </div>

        </section>


        {/* ====================================================
            TRANSLATION
        ==================================================== */}

        <section className="translation-panel">

          <div className="translation-header">

            <div>

              <div className="eyebrow">
                TRANSLATION
              </div>

              <h2>
                Communication output
              </h2>

            </div>

            <div className="translation-language">

              <span className="language-dot" />

              {targetLanguage}

            </div>

          </div>


          <div className="translation-content">

            {/* ENGLISH */}

            <div className="translation-column">

              <div className="translation-column-header">

                <span>
                  English
                </span>

                <span className="language-tag">
                  SOURCE
                </span>

              </div>

              <div
                className={
                  sentence
                    ? "translation-main"
                    : "translation-main muted"
                }
              >
                {sentence ||
                  "Waiting for a completed sentence..."}
              </div>

            </div>


            <div className="translation-divider" />


            {/* TARGET */}

            <div className="translation-column">

              <div className="translation-column-header">

                <span>
                  {targetLanguage}
                </span>

                <span className="language-tag target">
                  OUTPUT
                </span>

              </div>

              <div
                className={
                  translation
                    ? "translation-main translated"
                    : "translation-main muted"
                }
              >
                {translation ||
                  "Translation will appear here..."}
              </div>

            </div>

          </div>


          <div className="translation-footer">

            <span>
              Powered by SignFlow language engine
            </span>

            {translation && (

              <button
                className="speak-button"
                onClick={() =>
                  sendControl("enter")
                }
              >
                <span className="speaker-icon">
                  ♪
                </span>

                Play speech
              </button>

            )}

          </div>

        </section>


        {/* ====================================================
            CONTROLS
        ==================================================== */}

        <section className="controls-panel">

          <div className="controls-info">

            <div className="eyebrow">
              CONTROLS
            </div>

            <h2>
              Control recognition
            </h2>

            <p>
              Use the buttons or keyboard shortcuts
              to control the session.
            </p>

          </div>


          <div className="controls-buttons">

            <button
              className="control-button secondary"
              onClick={() =>
                sendControl("backspace")
              }
            >

              <span className="button-icon">
                ↶
              </span>

              <span className="button-text">

                <strong>
                  Undo
                </strong>

                <small>
                  Backspace
                </small>

              </span>

            </button>


            <button
              className="control-button secondary"
              onClick={clearInterface}
            >

              <span className="button-icon">
                ×
              </span>

              <span className="button-text">

                <strong>
                  Clear
                </strong>

                <small>
                  C
                </small>

              </span>

            </button>


            <button
              className="control-button primary"
              onClick={() =>
                sendControl("enter")
              }
            >

              <span className="button-icon">
                ▶
              </span>

              <span className="button-text">

                <strong>
                  Finish & Speak
                </strong>

                <small>
                  Enter
                </small>

              </span>

            </button>


            <button
              className="control-button danger"
              onClick={() =>
                sendControl("quit")
              }
            >

              <span className="button-icon">
                ■
              </span>

              <span className="button-text">

                <strong>
                  Stop
                </strong>

                <small>
                  Recognition
                </small>

              </span>

            </button>

          </div>

        </section>

      </main>


      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer className="app-footer">

        <div>
          <strong>
            SignFlow
          </strong>

          <span>
            AI-powered sign language communication
          </span>
        </div>

        <div className="keyboard-hints">

          <span>
            ENTER
          </span>

          <span>
            Finish
          </span>

          <span>
            BACKSPACE
          </span>

          <span>
            Undo
          </span>

          <span>
            C
          </span>

          <span>
            Clear
          </span>

        </div>

      </footer>

    </div>
  );
}

export default App;