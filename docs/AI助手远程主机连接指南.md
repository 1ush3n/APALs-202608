# AI 助手远程主机连接指南

> 本文供 Codex、Claude Code、Cursor、VS Code AI 助手及其他可执行本地 Shell 命令的智能体使用。
>
> 核心方案：AI 助手运行在本地，通过本机 `ssh.exe` 或 `ssh` 调用远程 Linux 主机。不要把密码、私钥、API 令牌或 `auth.json` 写入本文、仓库、命令行参数或聊天记录。

## 1. 两种远程工作模式

### 1.1 PowerShell/SSH 命令桥接（推荐，适用于本项目）

本地 AI 助手负责分析和决策，远程主机负责执行命令：

```text
本地 AI 助手
    ↓ PowerShell + ssh.exe
远程 Linux 主机
    ↓
训练、测试、日志、结果和 GPU
```

这种方式不要求远程主机登录 Codex，也不要求远程主机能访问 ChatGPT 登录页面。远程主机只需要：

- SSH 服务可用；
- 本地 SSH 密钥可用；
- 项目文件已存在；
- Python、Conda、CUDA 等运行环境已配置。

限制：本地 AI 助手不会自动把远程目录变成本地工作区。读取、修改和复制文件时，需要通过 `ssh`、`scp` 或远程命令完成。

### 1.2 Codex 官方远程连接

Codex 桌面端通过 SSH 启动远程 Codex 服务，并把远程项目注册为远程工作区。该模式需要远程主机安装并认证 Codex，且远程主机的登录 Shell 能找到 `codex`。

如果远程主机无法访问 `chatgpt.com` 或 `api.openai.com`，使用本文的 PowerShell/SSH 命令桥接模式。

## 2. 当前 AutoDL 连接配置

当前已验证的 AutoDL 实例示例：

```text
主机：connect.westb.seetacloud.com
端口：22218
用户：root
本机私钥：C:\Users\13575\.ssh\id_rsa
远程项目目录（当前已验证示例）：/root/autodl-tmp/APALs-202608-clean
远程 Codex：/usr/local/bin/codex
```

AutoDL 实例重启、更换地区或更换实例后，主机和端口可能变化。每次连接失败时，应以 AutoDL 控制台当前显示的 SSH 命令为准。

本次实际使用的是 Git 仓库的独立干净目录 `/root/autodl-tmp/APALs-202608-clean`，分支为 `codex/operation-node-scope-ablation`。不要假定旧目录仍然是当前工作目录；连接后先执行 `pwd`、`git status --short --branch` 和 `git rev-parse HEAD`。

本文不保存 AutoDL 密码。密码登录只用于首次验证或密钥配置，日常连接使用 SSH 密钥。

## 3. SSH 密钥准备

### 3.1 Windows 生成密钥

在 Windows PowerShell 执行：

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.ssh" | Out-Null
ssh-keygen -t rsa -b 4096 -f "$env:USERPROFILE\.ssh\id_rsa"
```

为了让后台 AI 助手可以自动连接，在密码短语提示处可以按两次回车；如果使用了密码短语，必须额外配置 Windows `ssh-agent`。

公钥路径：

```text
C:\Users\13575\.ssh\id_rsa.pub
```

查看公钥：

```powershell
Get-Content -Raw "$env:USERPROFILE\.ssh\id_rsa.pub"
```

只复制 `.pub` 文件内容。私钥 `id_rsa` 永远不能上传到远程主机、提交 Git 或发送给 AI 助手。

### 3.2 AutoDL 添加公钥

在 AutoDL 控制台进入：

```text
容器实例 → 设置密钥登录 → 添加 SSH 公钥
```

粘贴 `id_rsa.pub` 的完整单行内容并保存。

### 3.3 Windows SSH 配置

创建或编辑：

```powershell
notepad "$env:USERPROFILE\.ssh\config"
```

配置示例：

```sshconfig
Host autodl-apal
    HostName connect.westb.seetacloud.com
    Port 22218
    User root
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
```

文件名必须是 `config`，不能是 `config.txt`。

## 4. 连接测试

### 4.1 使用 SSH 别名

Windows PowerShell：

```powershell
ssh.exe -o BatchMode=yes -o ConnectTimeout=10 autodl-apal "hostname; id -un; pwd"
```

Linux/macOS：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 autodl-apal 'hostname; id -un; pwd'
```

预期结果应包含：

```text
远程容器主机名
root
/root
```

### 4.2 不使用 SSH 别名

Windows PowerShell：

```powershell
ssh.exe -o BatchMode=yes -o ConnectTimeout=10 -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "hostname; id -un; pwd"
```

Linux/macOS：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 -i ~/.ssh/id_rsa -p 22218 root@connect.westb.seetacloud.com 'hostname; id -un; pwd'
```

`BatchMode=yes` 会禁止密码交互。对于 AI 助手，这是必要的：密钥不可用时应立即失败，而不是一直等待输入密码。

## 5. 让 AI 助手通过 PowerShell 操作远程主机

### 5.1 通用 Windows 命令模板

```powershell
ssh.exe -o BatchMode=yes -o ConnectTimeout=10 -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "远程命令"
```

例如查看远程项目：

```powershell
ssh.exe -o BatchMode=yes -o ConnectTimeout=10 -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "cd /root/autodl-tmp/APALs-202608-clean && git status --short --branch"
```

检查远程工具：

```powershell
ssh.exe -o BatchMode=yes -o ConnectTimeout=10 -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com 'test -x /root/miniconda3/bin/python && /root/miniconda3/bin/python -c "import sys; print(sys.executable); print(sys.version)"; command -v conda; command -v codex; codex --version'
```

PowerShell 会处理本地双引号和变量展开。远程命令中需要保留 `$` 时，应使用外层单引号：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com 'printf "user=%s\n" "$USER"; pwd'
```

为避免转义错误，复杂命令优先拆成多条简单命令；训练启动命令优先写成一行。

### 5.2 远程环境检查

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com 'bash -ilc "source /root/miniconda3/etc/profile.d/conda.sh; conda activate base; command -v python; python -c \"import sys; print(sys.executable)\"; command -v codex; codex --version"'
```

检查 Conda 环境：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com 'source /root/miniconda3/etc/profile.d/conda.sh; conda activate base; python -c "import sys; print(sys.executable)"'
```

实际服务器没有 `rag_env`。非交互 SSH Shell 可能找不到 `python`，最稳妥的方式是直接使用 `/root/miniconda3/bin/python`；需要激活环境时先加载 `/root/miniconda3/etc/profile.d/conda.sh`，再执行 `conda activate base`。不要因为本机有 `rag_env` 就在服务器上使用同名环境。

检查 CUDA：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

无卡模式返回：

```text
No devices were found
```

属于正常结果，但不能在该实例上启动 GPU 训练。

有卡开机后再检查 PyTorch：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com '/root/miniconda3/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU-only\")"'
```

无卡模式实测为 `torch.cuda.is_available() == False`；有卡模式实测可见 NVIDIA GPU。不能仅凭 Conda 环境存在就判定 CUDA 可用。

## 6. 远程文件传输

### 6.1 从本地上传文件

```powershell
scp.exe -i "$env:USERPROFILE\.ssh\id_rsa" -P 22218 "D:\APAL-Dynamic-v4\configs\example.yaml" root@connect.westb.seetacloud.com:/root/autodl-tmp/APALs-202608-clean/configs/
```

### 6.2 从远程下载文件

```powershell
scp.exe -i "$env:USERPROFILE\.ssh\id_rsa" -P 22218 root@connect.westb.seetacloud.com:/root/autodl-tmp/APALs-202608-clean/results/summary.csv "D:\APAL-Dynamic-v4\results\summary.csv"
```

### 6.3 上传整个项目

```powershell
scp.exe -r -i "$env:USERPROFILE\.ssh\id_rsa" -P 22218 "D:\APAL-Dynamic-v4" root@connect.westb.seetacloud.com:/root/autodl-tmp/
```

大型数据集、缓存、虚拟环境和 `results/` 不应无选择地重复上传。优先使用 Git、AutoDL 文件存储或网盘同步数据。

## 7. 长时间训练

SSH 断开不应终止训练。AutoDL 官方建议使用 `screen` 或 `tmux`。

启动后台任务：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "screen -dmS apal_train bash -lc 'cd /root/autodl-tmp/APALs-202608-clean && /root/miniconda3/bin/python train.py > train.log 2>&1'"
```

查看任务：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "screen -ls"
```

查看最新日志：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "tail -n 100 /root/autodl-tmp/APALs-202608-clean/train.log"
```

进入任务：

```powershell
ssh.exe -t -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "screen -r apal_train"
```

退出 `screen` 但保持任务运行：

```text
Ctrl+A，然后按 D
```

结束任务前先确认目标：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "screen -S apal_train -X hardcopy /tmp/apal_train_screen.log"
```

### 7.1 SSH 断开、恢复与并行任务

实际验证中曾出现 SSH `BrokenPipeError`；这属于连接中断，不等于 Python 任务本身报错。重调度验证脚本支持 `--resume`，恢复前先检查输出目录和日志：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "cd /root/autodl-tmp/APALs-202608-clean && /root/miniconda3/bin/python scripts/evaluate_reschedule_manifest.py --help"
```

需要脱离 SSH 连接运行时，使用独立的 `nohup` 命令：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "cd /root/autodl-tmp/APALs-202608-clean && nohup /root/miniconda3/bin/python scripts/evaluate_reschedule_manifest.py experiment=reschedule_task_delay model_path=... manifest_path=... instance_ids=[real_680] output_dir=results/... > /tmp/reschedule_real_680.log 2>&1 < /dev/null &"
```

多个任务并行时，为每个任务使用独立的 SSH 命令、日志和输出目录；不要把多个 `cd ... && command & command &` 拼在一条复杂命令中。Shell 中 `&` 与 `&&` 的组合容易导致只有部分任务启动，且不利于定位失败。启动后用 `pgrep -af evaluate_reschedule`、`tail -f /tmp/reschedule_real_680.log` 和输出目录清单确认。

### 7.2 磁盘空间

AutoDL 的 `/root/autodl-tmp` 是实例本地盘，实际使用中可用空间曾低于 10 GB。下载 checkpoint、复制项目或启动多组验证前先检查：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "df -h /root/autodl-tmp; du -sh /root/autodl-tmp/APALs-202608-clean"
```

不要在未确认路径和哈希前清理旧结果；优先清理可重新生成的缓存，并确保输出目录有足够空间。

## 8. AutoDL 学术资源加速

AutoDL 官方学术加速主要针对：

- `github.com`；
- `githubusercontent.com`；
- `githubassets.com`；
- `huggingface.co`。

终端启用：

```bash
source /etc/network_turbo
```

检查代理：

```bash
env | grep -i proxy
```

该加速不等价于通用外网代理，不能假定它能访问 `chatgpt.com` 或 `api.openai.com`。不需要时关闭：

```bash
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```

## 9. AutoDL 有卡/无卡模式与关机

无卡/有卡模式的切换属于 AutoDL 控制台或其控制面能力，不能假定可以通过容器内 SSH 命令完成。SSH 可以关闭当前 Linux 容器/系统，但通常不能可靠地把无卡实例切换成 GPU 实例；切换卡模式时应在 AutoDL 控制台完成，并以控制台重新显示的 SSH 地址为准。

远程关机使用完整路径：

```powershell
ssh.exe -o BatchMode=yes -o ConnectTimeout=10 -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "/usr/bin/shutdown -h now"
```

SSH 随后出现 `Connection ... closed by remote host` 属于关机导致的正常现象。关机前必须确认没有仍需保留的训练、验证或下载任务；关机不会替代结果完整性检查。

## 10. Codex 登录与 PowerShell 桥接的区别

### PowerShell 桥接模式

远程主机不需要登录 Codex：

```text
本地 Codex 的账号和模型会话
    ↓
本地执行 ssh.exe
    ↓
AutoDL 只执行远程 Shell 命令
```

远程主机可以没有 `codex`。当前 AutoDL 虽然已经安装：

```text
/usr/local/bin/codex
codex-cli 0.148.0
```

但 PowerShell 桥接模式不会调用远程 Codex。

### Codex 官方远程连接模式

需要远程主机安装并认证 Codex：

```bash
codex login --device-auth
```

如果远程主机不能访问 OpenAI 登录服务，该模式可能无法完成登录。此时使用本文第 5 节的 PowerShell 桥接模式。

## 11. AI 助手执行远程任务的规则

每个 AI 助手开始远程工作前应遵守以下顺序：

1. 先确认目标主机、端口、用户和远程项目目录。
2. 先执行只读检查：`hostname`、`pwd`、`git status`、Python、Conda、CUDA。
3. 所有远程命令使用 SSH 密钥，不在命令中写密码。
4. 长任务使用 `screen` 或 `tmux`。
5. 修改前先查看文件或 Git 状态。
6. 删除、覆盖、清空结果目录前必须确认精确路径。
7. 结果文件通过 `scp` 下载到本地后再分析。
8. 不要把私钥、密码、API 令牌、`~/.codex/auth.json` 写入日志、文档或提交记录。
9. 不要把无卡实例误认为 GPU 训练实例。
10. 每次远程操作结束后报告：主机、目录、命令、结果和是否修改了文件。

## 12. 常见故障

### `Could not resolve hostname autodl-apal`

说明 SSH 别名没有被当前 Shell 读取。绕过别名，使用完整地址：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "hostname"
```

### `Permission denied (publickey,password)`

检查：

- AutoDL 是否已添加 `id_rsa.pub`；
- `IdentityFile` 是否指向正确文件；
- 主机和端口是否是当前实例的最新信息；
- 当前 SSH 进程是否有权限读取私钥。

详细诊断：

```powershell
ssh.exe -v -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "hostname"
```

### `codex: command not found`

检查远程登录 Shell：

```powershell
ssh.exe -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com 'bash -ilc "command -v codex"'
```

如果是 `/usr/local/bin/codex`，说明 Codex 已安装；如果为空，检查安装路径和登录 Shell 的 `PATH`。

### `nvidia-smi: No devices were found`

当前实例是无卡模式，或 GPU 没有挂载。到 AutoDL 控制台确认实例配置，不要通过修改 Python 代码解决。

### 训练因 SSH 断开停止

说明任务没有使用 `screen`、`tmux` 或 `nohup`。重新启动时使用第 7 节的后台命令，并优先利用任务自身的 `--resume` 继续未完成输出。

### `curl` 访问 `chatgpt.com` 超时

这通常是远程出站网络问题，不影响本地 PowerShell + SSH 桥接模式。优先检查：

```bash
curl -4 -I --connect-timeout 10 --max-time 30 https://chatgpt.com
curl -4 -I --connect-timeout 10 --max-time 30 https://api.openai.com
```

## 13. 最小连接验收

Windows PowerShell 执行：

```powershell
ssh.exe -o BatchMode=yes -o ConnectTimeout=10 -i "$env:USERPROFILE\.ssh\id_rsa" -p 22218 root@connect.westb.seetacloud.com "hostname; id -un; pwd; test -x /root/miniconda3/bin/python; /root/miniconda3/bin/python -c 'import torch; print(torch.cuda.is_available())'; git -C /root/autodl-tmp/APALs-202608-clean status --short --branch"
```

满足以下条件即表示 PowerShell 桥接可用：

- SSH 不要求输入密码；
- 能返回远程主机名；
- 用户为预期的远程用户；
- 项目目录存在；
- `git status` 能正常执行。
