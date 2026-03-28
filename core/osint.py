import cv2
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUTS_DIR, GALLERY_DIR


class FaceEmbedder:

    def __init__(self):
        model_path = Path(__file__).parent.parent / "models"
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.trained     = False
        self.gallery     = {}
        print("[PhantomEye] OSINT face embedder ready — OpenCV LBPH")

    def detect_faces(self, image: np.ndarray) -> list:
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces  = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            results.append({
                "bbox"     : (x, y, x+w, y+h),
                "roi_gray" : face_resized,
                "roi_color": image[y:y+h, x:x+w],
            })
        return results

    def compute_histogram(self, face_gray: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist(
            [face_gray], [0], None, [256], [0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def compare(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        score = cv2.compareHist(
            hist1.reshape(-1, 1).astype(np.float32),
            hist2.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        return round(float(score), 4)


class OSINTAudit:

    def __init__(self):
        self.embedder     = FaceEmbedder()
        self.gallery      = {}
        self.audit_log    = []
        GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        self._load_gallery()
        print(f"[PhantomEye] OSINT gallery loaded — {len(self.gallery)} persons")

    def _load_gallery(self):
        supported = [".jpg", ".jpeg", ".png"]
        for img_path in GALLERY_DIR.iterdir():
            if img_path.suffix.lower() not in supported:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = self.embedder.detect_faces(img)
            if faces:
                hist = self.embedder.compute_histogram(faces[0]["roi_gray"])
                self.gallery[img_path.stem] = {
                    "hist"  : hist,
                    "source": img_path.name,
                }

    def add_to_gallery(self, image: np.ndarray, person_id: str) -> bool:
        faces = self.embedder.detect_faces(image)
        if not faces:
            print(f"[OSINT] No face detected in image for ID: {person_id}")
            return False
        hist = self.embedder.compute_histogram(faces[0]["roi_gray"])
        self.gallery[person_id] = {
            "hist"  : hist,
            "source": "runtime_upload",
        }
        out_path = GALLERY_DIR / f"{person_id}.jpg"
        cv2.imwrite(str(out_path), faces[0]["roi_color"])
        print(f"[OSINT] Added to gallery: {person_id}")
        return True

    def audit(self, query_image: np.ndarray, query_id: str = "unknown") -> dict:
        faces = self.embedder.detect_faces(query_image)

        if not faces:
            return {
                "query_id"      : query_id,
                "face_detected" : False,
                "exposure_score": 0,
                "matches"       : [],
                "risk_level"    : "UNKNOWN",
                "message"       : "No face detected in query image.",
            }

        query_hist = self.embedder.compute_histogram(faces[0]["roi_gray"])

        matches = []
        for person_id, data in self.gallery.items():
            if person_id == query_id:
                continue
            score = self.embedder.compare(query_hist, data["hist"])
            if score > 0.4:
                matches.append({
                    "matched_id"  : person_id,
                    "confidence"  : round(score * 100, 1),
                    "source"      : data["source"],
                })

        matches.sort(key=lambda x: x["confidence"], reverse=True)
        top_matches = matches[:5]

        exposure_score = self._compute_exposure(top_matches)
        risk_level     = self._risk_level(exposure_score)

        result = {
            "query_id"      : query_id,
            "face_detected" : True,
            "exposure_score": exposure_score,
            "matches"       : top_matches,
            "risk_level"    : risk_level,
            "total_checked" : len(self.gallery),
            "message"       : self._message(risk_level, len(top_matches)),
        }

        self.audit_log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result"   : result,
        })

        return result

    def _compute_exposure(self, matches: list) -> int:
        if not matches:
            return 5
        top_conf = matches[0]["confidence"] if matches else 0
        count    = len(matches)
        score    = min(100, int(top_conf * 0.7 + count * 6))
        return score

    def _risk_level(self, score: int) -> str:
        if score >= 70:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def _message(self, risk: str, match_count: int) -> str:
        msgs = {
            "HIGH"  : f"Critical — {match_count} strong matches found. High digital footprint.",
            "MEDIUM": f"Moderate exposure — {match_count} partial matches detected.",
            "LOW"   : "Low exposure — minimal matches in reference gallery.",
        }
        return msgs.get(risk, "Audit complete.")

    def save_report(self, result: dict) -> Path:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        fname    = f"osint_report_{result['query_id']}_{int(time.time())}.json"
        out_path = OUTPUTS_DIR / fname
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[OSINT] Report saved: {out_path}")
        return out_path

    def visualize(self, query_image: np.ndarray, result: dict) -> np.ndarray:
        out   = query_image.copy()
        h, w  = out.shape[:2]

        risk_colors = {
            "HIGH"   : (0, 0, 255),
            "MEDIUM" : (0, 165, 255),
            "LOW"    : (0, 255, 100),
            "UNKNOWN": (128, 128, 128),
        }
        color = risk_colors.get(result["risk_level"], (128, 128, 128))

        cv2.rectangle(out, (0, 0), (w, 90), (0, 0, 0), -1)

        cv2.putText(out, "PhantomEye  OSINT Audit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 100), 1, cv2.LINE_AA)

        cv2.putText(out,
                    f"Risk: {result['risk_level']}   Score: {result['exposure_score']}/100",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    color, 2, cv2.LINE_AA)

        cv2.putText(out,
                    f"Matches: {len(result['matches'])}   Checked: {result['total_checked']}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(out, result["message"],
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        return out


def run_osint_demo(query_image_path: str):
    query_path = Path(query_image_path)
    if not query_path.exists():
        print(f"[ERROR] Image not found: {query_path}")
        return

    audit = OSINTAudit()

    query_img = cv2.imread(str(query_path))
    if query_img is None:
        print(f"[ERROR] Cannot read image: {query_path}")
        return

    print(f"\n[OSINT] Running audit on: {query_path.name}")
    print(f"[OSINT] Gallery size: {len(audit.gallery)} persons")

    result = audit.audit(query_img, query_id=query_path.stem)

    print(f"\n{'='*50}")
    print(f"  PHANTOMEYE OSINT AUDIT REPORT")
    print(f"{'='*50}")
    print(f"  Query       : {result['query_id']}")
    print(f"  Face found  : {result['face_detected']}")
    print(f"  Risk level  : {result['risk_level']}")
    print(f"  Score       : {result['exposure_score']} / 100")
    print(f"  Matches     : {len(result['matches'])}")
    print(f"  Message     : {result['message']}")

    if result["matches"]:
        print(f"\n  Top matches:")
        for m in result["matches"]:
            print(f"    - {m['matched_id']}  confidence: {m['confidence']}%")

    print(f"{'='*50}\n")

    audit.save_report(result)

    vis = audit.visualize(query_img, result)
    out_path = OUTPUTS_DIR / f"osint_{query_path.stem}.jpg"
    cv2.imwrite(str(out_path), vis)

    cv2.imshow("PhantomEye — OSINT Audit  [any key to close]", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"[OSINT] Visual saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core/osint.py <image_path>")
        print("Example: python core/osint.py data/gallery/person1.jpg")
    else:
        run_osint_demo(sys.argv[1])