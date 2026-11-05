import os
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from dotenv import load_dotenv

from agents import run_step, QAResult
from profiles import load_profiles, get_profile

load_dotenv()

app = Flask(__name__)
CORS(app)

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "domains")
PROFILES = load_profiles(PROFILES_DIR)


@app.route("/")
def home():
    return render_template("index.html")


@app.get("/domains")
def domains():
    """List available profiles for the frontend dropdown."""
    return jsonify({
        "success": True,
        "domains": [
            {
                "id": profile_id,
                "name": profile["name"],
                "description": profile.get("description", ""),
                "channels": profile.get("channels", []),
                "steps": profile.get("step_labels", {}),
            }
            for profile_id, profile in PROFILES.items()
        }
    })


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    language = data.get("language", "English")
    domain_id = data.get("domain", "fraud_awareness")

    if not topic:
        return jsonify({"success": False, "error": "Please provide a topic."}), 400
    if domain_id not in PROFILES:
        return jsonify({
            "success": False,
            "error": f"Unknown domain '{domain_id}'. Use GET /domains to see options."
        }), 400

    profile = get_profile(domain_id, PROFILES)
    state: Dict[str, Any] = {
        "topic": topic,
        "language": language,
        "channels": data.get("channels") or profile.get("channels", []),
        "qa_checklist": profile.get("qa_checklist", []),
    }
    attempts: List[Dict[str, Any]] = []

    try:
        state["research"] = run_step("research", profile, state)
        state["content"] = run_step("content", profile, state)
        # Optimization uses the most recent draft (initially the generated content).
        state["previous_attempt"] = state["content"]
        state["optimized"] = run_step("optimization", profile, state)

        # Initial QA + at most two optimization/QA revision cycles.
        qa: QAResult = run_step("quality", profile, state)
        attempts.append({
            "attempt": 1,
            "stage": "initial_qa",
            "passed": qa.passed,
            "score": qa.score,
            "issues": qa.issues,
            "content": qa.approved_content or state["optimized"],
        })

        revision_count = 0
        current_content = qa.approved_content or state["optimized"]

        while not qa.passed and revision_count < 2:
            revision_count += 1
            state["optimized"] = current_content
            state["qa_issues"] = qa.issues
            state["previous_attempt"] = current_content

            revised = run_step("optimization", profile, state)
            state["optimized"] = revised
            qa = run_step("quality", profile, state)
            current_content = qa.approved_content or revised

            attempts.append({
                "attempt": revision_count + 1,
                "stage": f"revision_{revision_count}",
                "status": "QA passed" if qa.passed else "QA rejected; revision completed",
                "passed": qa.passed,
                "score": qa.score,
                "issues": qa.issues,
                "content": current_content,
            })

        state["approved"] = current_content
        state["qa"] = qa.model_dump()
        state["final"] = run_step("distribution", profile, state)

        # Preserve the original response keys so the existing UI can still work.
        return jsonify({
            "success": True,
            "domain": domain_id,
            "domain_name": profile["name"],
            "research": state["research"],
            "content": state["content"],
            "optimized": state["optimized"],
            "approved": state["approved"],
            "qa": state["qa"],
            "attempts": attempts,
            "final": state["final"],
        })

    except Exception as exc:
        app.logger.exception("Generation failed")
        return jsonify({
            "success": False,
            "error": str(exc),
            "domain": domain_id,
            "attempts": attempts,
        }), 500


if __name__ == "__main__":
    # debug=False is safer outside local development.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
