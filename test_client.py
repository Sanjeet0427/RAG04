import json
import sys
import httpx

question = " ".join(sys.argv[1:]) or "What is the principle of least squares?"

with httpx.stream("POST", "http://127.0.0.1:8000/ask",
                  json={"question": question}, timeout=120) as r:
    event = None
    for line in r.iter_lines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if event == "sources":
                print("Sources:")
                for s in json.loads(data):
                    print(f"  p.{s['page']}: {s['preview']}")
                print("\nAnswer: ", end="", flush=True)
            elif event == "token":
                print(json.loads(data)["t"], end="", flush=True)
            elif event == "error":
                print("\nERROR:", data)
    print()