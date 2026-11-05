import os
import sys
import zipfile
import json
from datetime import datetime

zip_path = "/Users/heemanshupjain/Downloads/Member 2.zip"
print(f"Checking zip archive: {zip_path}")
print(f"Exists: {os.path.exists(zip_path)}")
stat_info = os.stat(zip_path)
print(f"File size on disk: {stat_info.st_size} bytes ({stat_info.st_size / (1024*1024):.2f} MB / {stat_info.st_size / (1024*1024*1024):.4f} GB)")
print(f"Modified: {datetime.fromtimestamp(stat_info.st_mtime).isoformat()}")

suspicious_entries = []
total_uncompressed = 0
total_files = 0
total_dirs = 0
extension_counts = {}
top_level_items = set()

with zipfile.ZipFile(zip_path, 'r') as zf:
    infolist = zf.infolist()
    print(f"Total entries in archive: {len(infolist)}")
    
    for info in infolist:
        name = info.filename
        
        # Check path traversal
        if name.startswith('/') or name.startswith('\\') or '..' in name.split('/') or '..' in name.split('\\'):
            suspicious_entries.append((name, "path traversal / relative upward / absolute"))
        if '\0' in name:
            suspicious_entries.append((name, "null byte"))
            
        parts = [p for p in name.replace('\\', '/').split('/') if p]
        if parts:
            top_level_items.add(parts[0])
            
        if info.is_dir():
            total_dirs += 1
        else:
            total_files += 1
            total_uncompressed += info.file_size
            ext = os.path.splitext(name)[1].lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1

print("\n--- Path Traversal & Security Check ---")
if suspicious_entries:
    print(f"WARNING: Found {len(suspicious_entries)} suspicious entries:")
    for s in suspicious_entries[:20]:
        print(f"  {s}")
else:
    print("Clean: No path traversal, no absolute paths, no null bytes detected in archive.")

print("\n--- Summary ---")
print(f"Total files: {total_files}")
print(f"Total directories: {total_dirs}")
print(f"Total uncompressed size: {total_uncompressed} bytes ({total_uncompressed / (1024*1024):.2f} MB / {total_uncompressed / (1024*1024*1024):.4f} GB)")
print(f"Top-level items in archive: {sorted(list(top_level_items))}")

print("\n--- Extensions breakdown ---")
for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(f"  {ext or '[no extension]'}: {count}")

# Check model/checkpoint files specifically in the zip
checkpoint_exts = {'.pt', '.pth', '.ckpt', '.onnx', '.safetensors', '.bin', '.h5'}
print("\n--- Checkpoint & Model files in zip ---")
for info in infolist:
    ext = os.path.splitext(info.filename)[1].lower()
    if ext in checkpoint_exts or 'model' in info.filename.lower() and ext in {'.pt', '.pth', '.ckpt'}:
        print(f"  {info.filename} ({info.file_size} bytes, compressed {info.compress_size} bytes)")

# Check largest 15 files in zip
print("\n--- 15 Largest Files in Archive ---")
sorted_by_size = sorted([info for info in infolist if not info.is_dir()], key=lambda x: x.file_size, reverse=True)[:15]
for info in sorted_by_size:
    print(f"  {info.filename}: {info.file_size / (1024*1024):.2f} MB ({info.file_size} bytes)")
