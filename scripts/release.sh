#!/bin/sh
# 打一个发版包：release 构建 → zipalign → 签名 → 生成 releases/app-version.json，releases/ 里只留最近 5 个 APK。
# 用法: scripts/release.sh <versionCode> <versionName> "更新说明"
# 签名用本机 ~/.android/debug.keystore（和 TV Player 一样）：换电脑要带上这个文件，否则新包装不上（签名不一致）。
set -e
[ $# -ge 3 ] || { echo "用法: $0 <versionCode> <versionName> \"更新说明\""; exit 1; }
CODE=$1
NAME=$2
NOTES=$3
cd "$(dirname "$0")/.."
BT=${ANDROID_BUILD_TOOLS:-/opt/homebrew/share/android-commandlinetools/build-tools/36.0.0}
OUT=releases/AI-Photo-$NAME.apk
URL=https://gh-proxy.com/https://raw.githubusercontent.com/photo2ai/ai-photo/main/$OUT

gradle assembleRelease -PappVersionCode="$CODE" -PappVersionName="$NAME" --console=plain -q
ALIGNED=$(mktemp -t aiphoto-aligned).apk
"$BT/zipalign" -f 4 app/build/outputs/apk/release/app-release-unsigned.apk "$ALIGNED"
# minSdk 26 用不到 v1（JAR）签名；v4 会多生成一个 .idsig 文件，也不要
"$BT/apksigner" sign --ks ~/.android/debug.keystore --ks-pass pass:android \
    --v1-signing-enabled false --v4-signing-enabled false --out "$OUT" "$ALIGNED"
rm -f "$ALIGNED"
"$BT/apksigner" verify --print-certs "$OUT" | grep "SHA-256"
python3 scripts/make-update-manifest.py "$OUT" "$URL" "$NOTES" > releases/app-version.json
ls -t releases/AI-Photo-*.apk | tail -n +6 | xargs rm -f
ls -l "$OUT"
cat releases/app-version.json
