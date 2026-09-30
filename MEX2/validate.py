"""Final end-to-end validation: import every module, load models, and run one
recorded clip per intent through the full classifier->dispatcher pipeline."""
import sys, os, glob, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("== Import check ==")
import voice_assistant
from voice_assistant import features, wakeword, classifier, actions, audio_input, assistant, ui
print("  all modules import OK")

print("== Model check ==")
from voice_assistant.classifier import CommandClassifier
from voice_assistant.wakeword import WakeWordDetector
from voice_assistant.assistant import Assistant
MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
a = Assistant(os.path.join(MD, "wakeword.joblib"), os.path.join(MD, "commands.joblib"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "music"), on_event=None)
print(f"  commands: {len(a.cmd.intents)} intents | wakeword loaded: {a.wake.pos_model is not None}")

print("== Full pipeline (one clip per intent) ==")
rows = list(csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.csv"))))
seen = {}
for r in rows:
    if r["intent"] not in seen and r["split"] == "train":
        seen[r["intent"]] = r["path"]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ok = miss = 0
for intent, path in sorted(seen.items()):
    fp = os.path.join(DATA, path.replace("/", os.sep))
    if not os.path.exists(fp):
        miss += 1
        continue
    x = features.load_audio(fp)
    res = a.handle_utterance(x)
    flag = "OK " if res["intent"] == intent else "MISS"
    if res["intent"] == intent: ok += 1
    else: miss += 1
    print(f"  [{flag}] {intent:16s} -> {res['intent']:16s}  {res['status'][:40]}")
print(f"\n  {ok} correct / {miss} misclassified (single-clip, no slot text)")
print("VALIDATION COMPLETE")
