from rag.retrieve import retrieve_evidence
import json

evidence = retrieve_evidence("Machine M3 vibration at 2.8x normal")
print(json.dumps(evidence, indent=2))
