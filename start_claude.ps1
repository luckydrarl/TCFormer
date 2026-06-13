# =========================================
# Claude Code 快速启动脚本 (PowerShell)
# 使用你现有 API Key
# =========================================

# 1️⃣ 临时绕过 PowerShell 执行策略
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 2️⃣ 设置 Claude Code 环境变量
$env:ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
$env:ANTHROPIC_AUTH_TOKEN="sk-79cec0fd38e34856b43a1516fe12fec2"  # 你的现有 API Key
$env:ANTHROPIC_MODEL="deepseek-v4-pro[1m]"
$env:ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-pro[1m]"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-pro[1m]"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
$env:CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
$env:CLAUDE_CODE_EFFORT_LEVEL="max"

# 3️⃣ 切换到项目目录
cd "C:\Users\SBW\Desktop\校智通ai"

# 4️⃣ 启动 Claude
claude.cmd