import os
import sys
import json
import requests
from rag.retrieve import retrieve_evidence

def diagnose_anomaly(anomaly_query: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set it before running: export GROQ_API_KEY='your_api_key'", file=sys.stderr)
        sys.exit(1)

    # 1. Retrieve context evidence
    print(f"Retrieving evidence for: '{anomaly_query}'...")
    evidence = retrieve_evidence(anomaly_query, n_results=2)

    # 2. Format context for prompt
    incidents_str = json.dumps(evidence.get("incidents", []), indent=2)
    sops_str = json.dumps(evidence.get("sops", []), indent=2)
    logs_str = json.dumps(evidence.get("maintenance_logs", []), indent=2)

    # 3. Define System and User Prompts
    system_prompt = """You are an expert Reliability Engineer and Maintenance Copilot. Your role is to analyze a machine anomaly report, evaluate the retrieved context (historical incidents, SOP rules, and maintenance logs), and provide a precise, structured, and action-oriented diagnosis.

Adhere to the following strict guidelines:
1. Grounding: Rely ONLY on the provided Context (Incidents, SOPs, and Maintenance Logs). Do not assume or extrapolate beyond the provided data. If the context does not contain enough information to make a definitive diagnosis, state this clearly.
2. SOP Adherence: Always prioritize Standard Operating Procedures (SOPs). If a retrieved SOP rule matches the anomaly condition (e.g. thresholds on vibration or temperature), highlight it immediately and state the required action.
3. Historical Correlation: Correlate the current anomaly with past incidents. Note if similar issues occurred on this machine or other machines, what their resolutions were, and their associated costs.
4. Maintenance Context: Check recent maintenance logs for the machine to see what actions were recently completed (e.g., replacements, inspections), who performed them, and their status. This helps identify if a recent repair was ineffective or if an inspection is overdue.
5. Tone: Be concise, analytical, objective, and professional.

Format your output in clear Markdown with the following sections:
- **CRITICAL ALERT / IMMEDIATE ACTIONS**: Required procedures based on SOP rules (if any).
- **ROOT CAUSE DIAGNOSIS**: Analysis of what is happening, correlating current anomaly with historical incidents and rules.
- **HISTORICAL CONTEXT**: Relevant past incidents and maintenance history for this machine.
- **RECOMMENDED NEXT STEPS**: Specific next steps for the technician, including who to consult if needed (referencing technicians from logs if relevant)."""

    user_prompt = f"""Anomaly Report:
"{anomaly_query}"

---

CONTEXT:
1. **Retrieved Historical Incidents**:
{incidents_str}

2. **Retrieved Standard Operating Procedures (SOPs)**:
{sops_str}

3. **Retrieved Maintenance Logs**:
{logs_str}

---

Please provide the diagnostic report following the system instructions."""

    # 4. Call Groq API (OpenAI-compatible)
    print("Calling Groq API for analysis...")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # We will use llama3-8b-8192 as a fast and capable model.
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1, # Low temperature for factual reliability diagnosis
        "max_tokens": 1024
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result_json = response.json()
        return result_json["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying Groq API: {e}", file=sys.stderr)
        if 'response' in locals() and response is not None:
            print(f"Response details: {response.text}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    query = "Machine M3 vibration at 2.8x normal"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    
    report = diagnose_anomaly(query)
    print("\n=== DIAGNOSTIC REPORT ===\n")
    print(report)
