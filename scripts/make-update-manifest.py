#!/usr/bin/env python3
"""输出发版 APK 对应的 app-version.json（应用内更新读它）。

用法: make-update-manifest.py <app.apk> <下载地址> ["更新说明"] > releases/app-version.json

versionCode / versionName 取自构建产物 app/build/outputs/apk/release/output-metadata.json，所以要先用同样的
-PappVersionCode / -PappVersionName 打 release 包。一般直接用 scripts/release.sh。
"""
import hashlib, json, os, sys

if len(sys.argv) < 3:
    sys.exit(__doc__)
apk, url = sys.argv[1], sys.argv[2]
notes = sys.argv[3] if len(sys.argv) > 3 else ""
meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "build", "outputs", "apk", "release", "output-metadata.json")
element = json.load(open(meta_path))["elements"][0]
data = open(apk, "rb").read()
print(json.dumps({
    "versionCode": element["versionCode"],
    "versionName": element["versionName"],
    "url": url,
    "sha256": hashlib.sha256(data).hexdigest(),
    "size": len(data),
    "notes": notes,
    "force": False,
}, ensure_ascii=False, indent=2))
