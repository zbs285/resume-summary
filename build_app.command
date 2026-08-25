#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]] || [[ "$($PYTHON_BIN -c 'import sys; print(int(sys.version_info >= (3, 11)))')" != "1" ]]; then
  echo "需要 Python 3.11 或更高版本。"
  exit 1
fi

if [[ ! -d .build-venv ]]; then
  "$PYTHON_BIN" -m venv .build-venv
fi
.build-venv/bin/python -m pip install -r requirements-dev.txt

PYINSTALLER_CONFIG_DIR="$PROJECT_DIR/work/pyinstaller-config" \
  .build-venv/bin/pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --onedir \
  --name ResumeSummary \
  --osx-bundle-identifier ai.openai.codex.resume-summary \
  --collect-all fitz \
  --collect-all reportlab \
  app_launcher.py

APP_PATH="$PROJECT_DIR/dist/ResumeSummary.app"
/usr/libexec/PlistBuddy -c 'Set :CFBundleDisplayName Resume Summary' "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :CFBundleName Resume Summary' "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :CFBundleShortVersionString 0.2.0' "$APP_PATH/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Add :CFBundleVersion string 0.2.0' "$APP_PATH/Contents/Info.plist" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c 'Set :CFBundleVersion 0.2.0' "$APP_PATH/Contents/Info.plist"
codesign --force --deep --sign - "$APP_PATH"
mv "$APP_PATH" "$PROJECT_DIR/dist/Resume Summary.app"

echo "构建完成：$PROJECT_DIR/dist/Resume Summary.app"

