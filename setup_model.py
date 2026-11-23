"""Restore the pinned model and Windows CUDA runtime after cloning the repository."""

import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen3-0.6B-Q8_0.gguf"
MODEL_URL = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/"
RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/b12110/"
ARCHIVES = {
    "llama-b12110-bin-win-cuda-12.4-x64.zip": "1ce755bc3a061889234048f88e33328e17971eaad73c4666cfeb26026b317fee",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
}
MODEL_SHA256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected):
    print(f"Downloading {destination.name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)
    if checksum(destination) != expected:
        raise ValueError(f"SHA-256 mismatch for {destination.name}")


def main():
    weights = ROOT / "model" / MODEL
    weights.parent.mkdir(exist_ok=True)
    engine = ROOT / "runtime" / "llama"
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        work = Path(temp)
        if not weights.exists() or checksum(weights) != MODEL_SHA256:
            download(MODEL_URL + MODEL, work / MODEL, MODEL_SHA256)
            (work / MODEL).replace(weights)
        for name, expected in ARCHIVES.items():
            archive = work / name
            download(RELEASE + name, archive, expected)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(work / "llama")
        if not (work / "llama" / "llama-server.exe").is_file():
            raise ValueError("The llama.cpp archive is missing its server executable")
        shutil.copytree(work / "llama", engine, dirs_exist_ok=True)
    print("Model and runtime are ready.")


if __name__ == "__main__":
    main()
