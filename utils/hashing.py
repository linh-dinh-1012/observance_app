import hashlib

def compute_file_hash(file_path, algo="sha256", block_size=65536):
    """Return hash from file contents"""
    if algo == "sha256":
        hasher = hashlib.sha256()
    elif algo == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError("Unsupported hash algorithm")
    
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()

