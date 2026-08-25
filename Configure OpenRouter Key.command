#!/bin/zsh
set -euo pipefail

SERVICE_NAME="Resume Summary OpenRouter API Key"

osascript -e 'display dialog "Resume Summary 使用 OpenRouter 当前可用的免费模型。请先创建一个 API Key；Key 只保存在本机 macOS 钥匙串。" buttons {"取消", "打开 API Key 页面"} default button "打开 API Key 页面" with title "配置 Resume Summary"'
open "https://openrouter.ai/settings/keys"

API_KEY="$(osascript -e 'text returned of (display dialog "创建后，把 API Key 粘贴到下面。" default answer "" with hidden answer buttons {"取消", "保存"} default button "保存" with title "保存 OpenRouter API Key")')"

if [[ -z "$API_KEY" ]]; then
  osascript -e 'display alert "API Key 不能为空" as critical'
  exit 1
fi

security add-generic-password -a "$USER" -s "$SERVICE_NAME" -w "$API_KEY" -U >/dev/null
osascript -e 'display notification "API Key 已保存到 macOS 钥匙串" with title "Resume Summary"'

