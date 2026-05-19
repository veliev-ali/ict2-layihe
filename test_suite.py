import io
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from app import app

PASS = 0
FAIL = 0


def ok(label, status, expected_status, body, check=None):
    global PASS, FAIL
    status_match = status == expected_status
    body_ok = True
    detail = ""
    if check and callable(check):
        body_ok, detail = check(body)

    if status_match and body_ok:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")
        if not status_match:
            print(f"         Expected HTTP {expected_status}, got {status}")
        if not body_ok:
            print(f"         {detail}")
        print(f"         Body: {body}")


def similarity_between(lo, hi):
    def check(body):
        s = body.get("similarity")
        if s is None:
            return False, "missing 'similarity' field"
        if not (lo <= s <= hi):
            return False, f"expected similarity in [{lo}, {hi}], got {s}"
        return True, ""
    return check


def has_error(substring):
    def check(body):
        err = body.get("error", "")
        if substring not in err:
            return False, f"expected error containing '{substring}', got '{err}'"
        return True, ""
    return check


def valid_number(body):
    s = body.get("similarity")
    if s is None:
        return False, "missing similarity"
    if not isinstance(s, (int, float)):
        return False, f"not a number: {type(s)}"
    if s < 0 or s > 100:
        return False, f"out of range: {s}"
    if round(s, 2) != s:
        return False, f"more than 2 decimal places: {s}"
    return True, ""


# -----------------------------------------------------------------------
# TEXT
# -----------------------------------------------------------------------
print("\n=== TEXT ===")

def t(t1, t2, fn1="a.txt", fn2="b.txt"):
    with app.test_client() as c:
        return c.post("/compare", data={
            "file1": (io.BytesIO(t1.encode() if isinstance(t1, str) else t1), fn1),
            "file2": (io.BytesIO(t2.encode() if isinstance(t2, str) else t2), fn2),
            "modality": "text",
        })

status, body = (lambda r: (r.status_code, r.get_json()))(t("hello world", "hello world"))
ok("identical", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(t("the quick brown fox jumps", "the quick brown fox leaps"))
ok("near-identical", status, 200, body, similarity_between(60, 95))

status, body = (lambda r: (r.status_code, r.get_json()))(t("dog cat bird fish", "dog cat bird lizard"))
ok("shared vocabulary", status, 200, body, similarity_between(30, 90))

status, body = (lambda r: (r.status_code, r.get_json()))(t("apple banana cherry", "xylophone zebra quantum"))
ok("disjoint vocabulary (char n-grams may overlap slightly)", status, 200, body, similarity_between(0, 15))

status, body = (lambda r: (r.status_code, r.get_json()))(t("hello world", "hello world foo bar baz qux"))
ok("subset", status, 200, body, similarity_between(0, 95))

status, body = (lambda r: (r.status_code, r.get_json()))(t("data data data data", "data"))
ok("single word repeated", status, 200, body, similarity_between(90, 100))

status, body = (lambda r: (r.status_code, r.get_json()))(t("lorem ipsum " * 200, "lorem ipsum " * 200))
ok("long identical (5KB)", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(t("the " * 500 + "alpha", "the " * 500 + "beta"))
ok("long near-identical", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(t("caf\u00e9 r\u00e9sum\u00e9", "caf\u00e9 r\u00e9sum\u00e9"))
ok("unicode identical", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(t("def foo(): return 42", "def bar(): return 99"))
ok("code-like text", status, 200, body, similarity_between(30, 80))

status, body = (lambda r: (r.status_code, r.get_json()))(t("   \n\t  ", "   \n\t  "))
ok("whitespace only", status, 200, body, similarity_between(0, 0.1))

status, body = (lambda r: (r.status_code, r.get_json()))(t("", "some text here"))
ok("one empty", status, 200, body, similarity_between(0, 0.1))

status, body = (lambda r: (r.status_code, r.get_json()))(t("", ""))
ok("both empty", status, 200, body, similarity_between(0, 0.1))

status, body = (lambda r: (r.status_code, r.get_json()))(t("Hello World", "hello world"))
ok("case differences (TF-IDF lowercases by default)", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(t("Hello!!! How are you???", "Hello... How are you???"))
ok("punctuation heavy (char n-grams include punct)", status, 200, body, similarity_between(40, 90))

status, body = (lambda r: (r.status_code, r.get_json()))(t("data science machine learning", "machine learning data science"))
ok("same words different order (TF-IDF ignores order)", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(t(b"\x00\x01\x02\xff\xfe", "hello"))
ok("binary junk in text mode", status, 200, body, similarity_between(0, 0.1))


# -----------------------------------------------------------------------
# IMAGE
# -----------------------------------------------------------------------
print("\n=== IMAGE ===")

from PIL import Image
import numpy as np


def png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def img_post(d1, d2, fn1="a.png", fn2="b.png"):
    with app.test_client() as c:
        return c.post("/compare", data={
            "file1": (io.BytesIO(d1), fn1),
            "file2": (io.BytesIO(d2), fn2),
            "modality": "image",
        })


black = png(Image.new("L", (64, 64), 0))
white = png(Image.new("L", (64, 64), 255))
gray = png(Image.new("L", (64, 64), 128))
dark = png(Image.new("L", (64, 64), 10))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(black, black))
ok("identical solid", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(black, white))
ok("black vs white", status, 200, body, similarity_between(0, 0.1))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(black, dark))
ok("black vs dark gray (zero-norm black → 0%)", status, 200, body, similarity_between(0, 0.1))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(gray, white))
ok("gray vs white", status, 200, body, similarity_between(98, 100.0))


def gradient(roll=0):
    arr = np.tile(np.roll(np.linspace(0, 255, 64, dtype=np.uint8), roll), (64, 1))
    return png(Image.fromarray(arr, mode="L"))


g0 = gradient(0)
g5 = gradient(5)
g30 = gradient(30)

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(g0, gradient(0)))
ok("identical gradients", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(g0, g5))
ok("slightly shifted gradient", status, 200, body, similarity_between(85, 99.9))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(g0, g30))
ok("more shifted gradient", status, 200, body, similarity_between(50, 95))

# different sizes, same fill
tiny = png(Image.new("L", (16, 16), 128))
large = png(Image.new("L", (256, 256), 128))
status, body = (lambda r: (r.status_code, r.get_json()))(img_post(tiny, large))
ok("different sizes same fill", status, 200, body, similarity_between(99.9, 100.0))

# non-square
wide = png(Image.new("L", (320, 160), 200))
tall = png(Image.new("L", (160, 320), 200))
status, body = (lambda r: (r.status_code, r.get_json()))(img_post(wide, tall))
ok("different aspect ratios same fill", status, 200, body, similarity_between(99.9, 100.0))

# 1x1 pixel
p1 = png(Image.new("L", (1, 1), 42))
status, body = (lambda r: (r.status_code, r.get_json()))(img_post(p1, png(Image.new("L", (1, 1), 42))))
ok("1x1 identical", status, 200, body, similarity_between(99.9, 100.0))

# checkerboard
def checkerboard(sq=8):
    arr = np.zeros((64, 64), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            if ((i // sq) + (j // sq)) % 2 == 0:
                arr[i, j] = 255
    return png(Image.fromarray(arr, mode="L"))

cb8 = checkerboard(8)
cb16 = checkerboard(16)
status, body = (lambda r: (r.status_code, r.get_json()))(img_post(cb8, checkerboard(8)))
ok("identical checkerboard", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(cb8, cb16))
ok("different checkerboard densities", status, 200, body, similarity_between(50, 99.9))

# random noise
np.random.seed(1)
r1 = png(Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L"))
np.random.seed(2)
r2 = png(Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L"))
np.random.seed(1)
r1b = png(Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8), mode="L"))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(r1, r1b))
ok("random noise same seed", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(r1, r2))
ok("random noise different seeds", status, 200, body, similarity_between(0, 99))

# JPEG vs PNG (approximate due to lossy compression)
bw = Image.new("L", (64, 64), 77)
rgb = bw.convert("RGB")
jbuf = io.BytesIO()
rgb.save(jbuf, format="JPEG", quality=95)
jpg_data = jbuf.getvalue()

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(png(bw), jpg_data, "a.png", "b.jpg"))
ok("PNG vs JPEG same content", status, 200, body, similarity_between(85, 100.0))

# partially different — half black, half white vs half white, half black
def half_black_top():
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[:32, :] = 255
    return png(Image.fromarray(arr, mode="L"))

def half_black_bottom():
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[32:, :] = 255
    return png(Image.fromarray(arr, mode="L"))

status, body = (lambda r: (r.status_code, r.get_json()))(img_post(half_black_top(), half_black_bottom()))
ok("half top-white vs half bottom-white", status, 200, body, similarity_between(0, 1))


# -----------------------------------------------------------------------
# AUDIO
# -----------------------------------------------------------------------
print("\n=== AUDIO ===")

import soundfile as sf


def wav(samples, sr):
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def aud_post(d1, d2, fn1="a.wav", fn2="b.wav"):
    with app.test_client() as c:
        return c.post("/compare", data={
            "file1": (io.BytesIO(d1), fn1),
            "file2": (io.BytesIO(d2), fn2),
            "modality": "audio",
        })


SR = 22050
DUR = 1.5
time_axis = np.linspace(0, DUR, int(SR * DUR), endpoint=False).astype(np.float32)

tone_440 = np.sin(2 * np.pi * 440 * time_axis).astype(np.float32)
tone_880 = np.sin(2 * np.pi * 880 * time_axis).astype(np.float32)
tone_220 = np.sin(2 * np.pi * 220 * time_axis).astype(np.float32)
tone_1k  = np.sin(2 * np.pi * 1000 * time_axis).astype(np.float32)
chord_raw = np.sin(2 * np.pi * 440 * time_axis) + np.sin(2 * np.pi * 554 * time_axis) + np.sin(2 * np.pi * 659 * time_axis)
chord = (chord_raw / np.max(np.abs(chord_raw))).astype(np.float32)
silence = np.zeros_like(tone_440, dtype=np.float32)
quiet = (tone_440 * 0.01).astype(np.float32)

np.random.seed(42)
noise1 = (np.random.randn(len(time_axis)) * 0.3).astype(np.float32)
np.random.seed(99)
noise2 = (np.random.randn(len(time_axis)) * 0.3).astype(np.float32)

w_440, w_880, w_220, w_1k = wav(tone_440, SR), wav(tone_880, SR), wav(tone_220, SR), wav(tone_1k, SR)
w_chord = wav(chord, SR)
w_silence = wav(silence, SR)
w_quiet = wav(quiet, SR)
w_noise1, w_noise2 = wav(noise1, SR), wav(noise2, SR)

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, wav(tone_440, SR)))
ok("identical tone", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_noise1, wav(noise1, SR)))
ok("identical noise", status, 200, body, similarity_between(99.9, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, w_quiet))
ok("same tone, different amplitude", status, 200, body, similarity_between(98, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, w_880))
ok("440Hz vs 880Hz", status, 200, body, similarity_between(0, 99.9))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, w_220))
ok("440Hz vs 220Hz", status, 200, body, similarity_between(0, 99.9))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_220, w_1k))
ok("220Hz vs 1000Hz", status, 200, body, similarity_between(0, 99.9))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, w_chord))
ok("pure tone vs chord", status, 200, body, similarity_between(0, 99.9))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, w_noise1))
ok("tone vs white noise", status, 200, body, similarity_between(0, 50))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_noise1, w_noise2))
ok("different white noise (similar flat MFCC)", status, 200, body, similarity_between(80, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_silence, w_440))
ok("silence vs tone (MFCC handles silence)", status, 200, body, similarity_between(80, 100.0))

status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_silence, w_silence))
ok("silence vs silence (both zero-norm → 100%)", status, 200, body, similarity_between(99.9, 100.0))

# different sample rates
SR8 = 8000
t8 = np.linspace(0, 1.0, SR8, endpoint=False).astype(np.float32)
tone_440_8k = np.sin(2 * np.pi * 440 * t8).astype(np.float32)
w_440_8k = wav(tone_440_8k, SR8)
status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440_8k, w_440))
ok("same tone, 8k vs 22k sample rate", status, 200, body, similarity_between(50, 100.0))

# different durations
ts = np.linspace(0, 0.2, int(SR * 0.2), endpoint=False).astype(np.float32)
tl = np.linspace(0, 3.0, int(SR * 3.0), endpoint=False).astype(np.float32)
w_440_short = wav(np.sin(2 * np.pi * 440 * ts).astype(np.float32), SR)
w_440_long  = wav(np.sin(2 * np.pi * 440 * tl).astype(np.float32), SR)
status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440_short, w_440_long))
ok("same tone, 0.2s vs 3.0s", status, 200, body, similarity_between(90, 100.0))

# FLAC test
flac_buf = io.BytesIO()
sf.write(flac_buf, tone_440, SR, format="FLAC")
w_440_flac = flac_buf.getvalue()
status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, w_440_flac, "a.wav", "b.flac"))
ok("WAV vs FLAC same tone", status, 200, body, similarity_between(99.9, 100.0))

# stereo mixed to mono
stereo = np.column_stack([tone_440, tone_440 * 0.5]).astype(np.float32)
w_stereo = wav(stereo, SR)
status, body = (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, w_stereo))
ok("mono vs stereo (mixed down)", status, 200, body, similarity_between(95, 100.0))


# -----------------------------------------------------------------------
# ERRORS
# -----------------------------------------------------------------------
print("\n=== ERRORS ===")

def err(data):
    with app.test_client() as c:
        return c.post("/compare", data=data)

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b"a"), "a.txt"),
    "file2": (io.BytesIO(b"b"), "b.txt"),
}))
ok("missing modality", status, 400, body, has_error("Invalid modality"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b"a"), "a.txt"),
    "file2": (io.BytesIO(b"b"), "b.txt"),
    "modality": "video",
}))
ok("invalid modality value", status, 400, body, has_error("Invalid modality"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file2": (io.BytesIO(b"b"), "b.txt"),
    "modality": "text",
}))
ok("missing file1", status, 400, body, has_error("Both files are required"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b"a"), "a.txt"),
    "modality": "text",
}))
ok("missing file2", status, 400, body, has_error("Both files are required"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b"a"), ""),
    "file2": (io.BytesIO(b"b"), "b.txt"),
    "modality": "text",
}))
ok("empty filename", status, 400, body, has_error("Both files must be selected"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({"modality": "text"}))
ok("no files at all", status, 400, body, has_error("Both files are required"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({}))
ok("empty POST body", status, 400, body, has_error("Both files are required"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(black), "b.png"),
    "modality": "image",
}))
ok("image — missing file2", status, 400, body, has_error("Both files are required"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b"not an image"), "a.png"),
    "file2": (io.BytesIO(black), "b.png"),
    "modality": "image",
}))
ok("image — garbage file", status, 422, body, has_error("Processing failed"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b""), "a.png"),
    "file2": (io.BytesIO(black), "b.png"),
    "modality": "image",
}))
ok("image — empty file", status, 422, body, has_error("Processing failed"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b"not audio"), "a.wav"),
    "file2": (io.BytesIO(w_440), "b.wav"),
    "modality": "audio",
}))
ok("audio — garbage file", status, 422, body, has_error("Processing failed"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b""), "a.wav"),
    "file2": (io.BytesIO(w_440), "b.wav"),
    "modality": "audio",
}))
ok("audio — empty file", status, 422, body, has_error("Processing failed"))

# wrong extension is OK if contents decode
status, body = (lambda r: (r.status_code, r.get_json()))(err({
    "file1": (io.BytesIO(b"hello world"), "a.png"),
    "file2": (io.BytesIO(b"hello world"), "b.png"),
    "modality": "text",
}))
ok("text with .png extension still works", status, 200, body, similarity_between(99.9, 100.0))


# -----------------------------------------------------------------------
# STRUCTURAL / CONTRACT TESTS
# -----------------------------------------------------------------------
print("\n=== CONTRACT ===")

status, body = (lambda r: (r.status_code, r.get_json()))(t("a", "a"))
ok("success response has only 'similarity' key", status, 200, body,
   lambda b: (set(b.keys()) == {"similarity"}, f"unexpected keys: {set(b.keys())}"))

status, body = (lambda r: (r.status_code, r.get_json()))(err({"modality": "text"}))
ok("error response has only 'error' key", status, 400, body,
   lambda b: (set(b.keys()) == {"error"}, f"unexpected keys: {set(b.keys())}"))

# Valid number range + rounding for each modality
for label, (st, bd) in [
    ("text", (lambda r: (r.status_code, r.get_json()))(t("hi", "hi"))),
    ("image", (lambda r: (r.status_code, r.get_json()))(img_post(gray, gray))),
    ("audio", (lambda r: (r.status_code, r.get_json()))(aud_post(w_440, w_440))),
]:
    ok(f"valid number [0,100] with <=2 decimals — {label}", st, 200, bd, valid_number)


# -----------------------------------------------------------------------
# SYMMETRY
# -----------------------------------------------------------------------
print("\n=== SYMMETRY ===")

with app.test_client() as c:
    res = c.post("/compare", data={
        "file1": (io.BytesIO(b"alpha beta gamma"), "a.txt"),
        "file2": (io.BytesIO(b"alpha beta delta"), "b.txt"),
        "modality": "text",
    })
    s1 = res.get_json()["similarity"]
    res = c.post("/compare", data={
        "file1": (io.BytesIO(b"alpha beta delta"), "a.txt"),
        "file2": (io.BytesIO(b"alpha beta gamma"), "b.txt"),
        "modality": "text",
    })
    s2 = res.get_json()["similarity"]
    diff = abs(s1 - s2)
    if diff < 0.01:
        PASS += 1
        print(f"  [PASS] symmetry — text ({s1} vs {s2})")
    else:
        FAIL += 1
        print(f"  [FAIL] symmetry — text: {s1} vs {s2}, diff={diff}")

    res = c.post("/compare", data={
        "file1": (io.BytesIO(black), "a.png"),
        "file2": (io.BytesIO(white), "b.png"),
        "modality": "image",
    })
    s1 = res.get_json()["similarity"]
    res = c.post("/compare", data={
        "file1": (io.BytesIO(white), "a.png"),
        "file2": (io.BytesIO(black), "b.png"),
        "modality": "image",
    })
    s2 = res.get_json()["similarity"]
    diff = abs(s1 - s2)
    if diff < 0.01:
        PASS += 1
        print(f"  [PASS] symmetry — image ({s1} vs {s2})")
    else:
        FAIL += 1
        print(f"  [FAIL] symmetry — image: {s1} vs {s2}, diff={diff}")

    res = c.post("/compare", data={
        "file1": (io.BytesIO(w_440), "a.wav"),
        "file2": (io.BytesIO(w_880), "b.wav"),
        "modality": "audio",
    })
    s1 = res.get_json()["similarity"]
    res = c.post("/compare", data={
        "file1": (io.BytesIO(w_880), "a.wav"),
        "file2": (io.BytesIO(w_440), "b.wav"),
        "modality": "audio",
    })
    s2 = res.get_json()["similarity"]
    diff = abs(s1 - s2)
    if diff < 0.01:
        PASS += 1
        print(f"  [PASS] symmetry — audio ({s1} vs {s2})")
    else:
        FAIL += 1
        print(f"  [FAIL] symmetry — audio: {s1} vs {s2}, diff={diff}")


# -----------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"{FAIL} TEST(S) FAILED")
    sys.exit(1)
