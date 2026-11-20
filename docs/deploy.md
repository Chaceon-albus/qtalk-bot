# 部署说明

写给运维者：在一台 Debian 的 VPS 上用 docker compose 运行 qtalk-bot。设计上的取舍见 [docs/design/deployment.md](design/deployment.md)。

机器人只往外连（QQ 开放平台、大模型服务），不开任何端口。一台 2 核 2G 的机器够用。

## 1. 主机

- **全盘加密**：主机要开全盘加密（云服务器用服务商提供的加密卷），数据目录放在加密的卷上。数据库里有用户的 openid、聊天内容与记忆，程序本身不加密数据库：机器人运行时要读它，密钥只能和数据库放在同一台机器上，挡不住主机被攻破；全盘加密防的是磁盘或快照单独外泄。
- **装 docker**：按 docker 的官方文档装 Docker Engine 与 compose 插件：<https://docs.docker.com/engine/install/debian/>。装好后 `docker compose version` 能输出版本即可。

## 2. 取代码

```sh
git clone <本仓库的地址> qtalk-bot
cd qtalk-bot
```

之后的命令都在这个目录里执行。

## 3. 准备 `.env` 与 `config.toml`

```sh
install -m 600 .env.example .env
install -m 600 config.example.toml config.toml   # 填之前就收紧：代理地址里可能带账号密码
```

- `.env` 只放密钥：`APP_ID`、`APP_SECRET`、`ADMIN_OPENID` 必填，`OPENAI_API_KEY`、`GEMINI_API_KEY` 用到哪家填哪家。compose 把它们逐项注入容器（见 `compose.yaml` 的 `environment:`），不会进镜像。`.env` 不要提交，也不要复制到别处。
- `config.toml` 放其余的设置，说明都在文件的注释里。容器里的配置路径固定是 `/app/config.toml`，**`data_dir` 保持 `"data"`**（它按配置文件所在的目录解析成 `/app/data`，正好是挂进来的 `./data`）。
- 告警与备份的阈值在 `[backup]`、`[alerts]` 两个表里，不写就用默认值：每 24 小时备份一次、留 7 份；模型连续失败 5 次、磁盘剩余不到 1024 MB、QQ 网关断开超过 10 分钟时告警。

`config.toml` 不放密钥，但 `[qq].proxy`、`[providers.*].proxy` 的地址里可以带用户名和密码，所以不能让宿主机上的其他账户读到它。做法是属主仍是你（运维者，方便编辑），组设成容器账户的 gid（`compose.yaml` 里 `build.args` 的 `GID`，默认 1000），权限 640，容器经组读它：

```sh
sudo chgrp 1000 config.toml   # 换成实际的 GID
chmod 640 config.toml
ls -l config.toml             # 应是 -rw-r----- 你 1000
```

有的编辑器保存时会另写一个新文件再改名，组和权限可能被重置，改完后用 `ls -l` 确认一下。`.env` 只由 compose 在宿主机上读，保持 600、属于你即可。

## 4. 数据目录的属主

容器里以 uid、gid 1000 的非 root 账户运行，挂进去的 `data/` 要属于它：

```sh
mkdir -p data
sudo chown 1000:1000 data
chmod 700 data
```

宿主机上想用别的 uid 时，改 `compose.yaml` 里 `build.args` 的 `UID`、`GID`，再按新的 uid 改 `data/` 的属主。

## 5. 启动与看日志

```sh
docker compose up -d --build   # 构建镜像并在后台启动
docker compose logs -f bot     # 看日志，Ctrl+C 退出（不影响运行）
docker compose stop            # 停止（程序会先收尾，最多 30 秒）
docker compose start           # 再启动
```

- 第一次启动后，管理员（`.env` 里的 `ADMIN_OPENID`）给机器人发 `/help` 可以看到管理命令。
- 日志只写到标准输出，由 docker 的 json-file 驱动轮转：单个文件 10 MB，保留 5 个。默认不记录消息正文。
- 重启策略是 `unless-stopped`：进程退出（包括网关无法恢复时）会被自动拉起，手动 `stop` 之后不会。
- 配置或密钥有误时进程会打印原因并退出，`docker compose logs bot` 里能看到；改好后 `docker compose up -d` 即可。

## 6. 看健康状态

```sh
docker compose ps    # STATUS 一列里有 (healthy) / (unhealthy) / (health: starting)
docker inspect --format '{{json .State.Health}}' "$(docker compose ps -q bot)"
```

健康检查每 30 秒运行一次 `python -m qtalk_bot.health`，连续 3 次失败显示为不健康（启动后有 60 秒的宽限）。主程序每 30 秒确认一次数据库能用，写一次心跳文件 `data/heartbeat.json`，里面带着当时 QQ 网关（收消息的长连接）的状态。下面任何一条成立就算这次失败：

- 心跳超过 90 秒没更新（进程卡住或已经退出），或心跳的时间比现在晚 30 秒以上（主机时钟被往回调过）。
- 网关断开超过 `[alerts].gateway_down_minutes`（默认 10 分钟）。
- 启动后超过同样的分钟数还没开始连网关（卡在启动阶段，比如一直取不到机器人的信息）。
- 网关已经停止（程序正在退出）。

网关连着、或断开还没到门槛时只看心跳。旧版本写的心跳文件里没有网关的部分，这时网关按未知处理，不算失败。

docker **不会**因为不健康而重启容器，这个状态只供查看与外部监控使用。看到不健康时先看日志，必要时 `docker compose restart bot`。

## 7. 给管理员的告警

以下情况机器人会主动给管理员发 QQ 消息，同一件事 24 小时内只发一次，恢复后补一条「已恢复」：

- 模型鉴权失败（401、403）或 OpenAI 额度用尽：立即发。
- 同一个模型连续失败 `[alerts].model_failures` 次（连接失败、超时、限流、服务端出错等）。
- 定时备份失败（1 小时后会再试）。
- 清理旧备份失败（比如某份超龄的备份删不掉）：新的备份照常做，清理 1 小时后再试；超龄的备份里可能有用户要求忘掉的内容，请检查 `data/backups/` 的权限。
- 数据目录所在的磁盘剩余空间低于 `[alerts].disk_free_min_mb`。
- QQ 网关断开超过 `[alerts].gateway_down_minutes`（默认 10 分钟），写明从什么时候（按配置的时区）断开；重新连上后补一条「已恢复」，写明断开了多久。程序每 30 秒看一次，期间断开又连上的不会告警。告警经 QQ 的 HTTP 接口发，网关断开时它多半仍然可用；也发不出去时只记日志。

启动后 5 分钟内不查磁盘、不补做备份，所以崩溃重启的循环不会每次都发一遍。告警是主动消息，管理员要保持「允许主动发送」。

**已知的边界**：网关遇到无法恢复的关闭码（例如凭证失效、被平台封禁）时，pyqbot 会让程序退出，容器由重启策略反复拉起。每次都活不到 10 分钟，所以**不会有网关告警**，健康检查也可能来不及变成不健康（每次重启都重新计时）。这时要看 `docker compose ps`（或 `docker ps`）里容器的重启次数与 `docker compose logs bot` 的报错。程序不另做持久化的告警。

## 8. 升级

```sh
git pull
docker compose up -d --build
```

- 新版本要升级表结构时，程序会先把数据库备份成 `data/backups/qtalk-before-v<旧版本>-<时间>.db`（最多留最近 3 份），备份失败就不升级、直接退出，日志里写明原因。
- 新版本增删了配置项时，对照 `config.example.toml` 修改 `config.toml`；出现程序不认识的键会报错退出。
- **升级 pyqbot**：pyqbot 从 GitHub 安装，固定到 `pyproject.toml` 里 `[tool.uv.sources]` 写的 commit。要用 pyqbot 的新版本时，在开发机上把 `rev` 改成新的 commit，运行 `uv lock --upgrade-package pyqbot` 更新 `uv.lock`，测试通过后提交、推送；服务器上照常 `git pull` 再 `docker compose up -d --build`。不要在服务器上手改这两个文件。
- 镜像只在 `--build` 时重建。基础镜像有安全更新时，可以 `docker compose build --pull` 再 `docker compose up -d`。

## 9. 备份与恢复

备份在 `data/backups/` 下（目录权限 0700，文件 0600）：

- 定时备份：`qtalk-YYYYMMDDTHHMMSSZ.db`（UTC 时间），启动 5 分钟后看最新的一份，超过 `[backup].interval_hours` 就做一份。最多留 `[backup].keep` 份，而且最老的不超过 `keep × interval_hours`（默认 7 × 24 小时，即 7 天）：超龄的即使新备份一直失败也会删掉。
- 迁移前备份：`qtalk-before-v<旧版本>-<时间>.db`，最多留 3 份，不计入上面的份数，也不受上面的年龄上限约束。
- 剩余磁盘空间不到数据库大小的 2 倍时不做备份，按失败告警。
- 程序只轮转符合上面命名的文件，你自己放进这个目录的文件不会被删掉。进程被强杀时没做完的备份会留下 `.qtalk-…partial` 临时文件，下次启动时自动清掉。
- 备份里可能还有用户已经要求忘掉的内容，定时备份最多再留 `keep × interval_hours`（默认 7 天）；迁移前备份不受这个上限约束，升级完确认无误后可以自己删掉。

**从备份恢复**：把下面整段复制到终端里执行（在仓库目录里，先把 `name=` 那一行换成要恢复的那一份的文件名）。整段在一个开了 `set -euo pipefail` 的子 shell 里跑：任何一步失败（备份不存在、读不了、校验不过、磁盘写满、停机失败……）都会立刻停下，不会启动容器，也不会在原位置留下一个空库。

```sh
(
  set -euo pipefail
  name=qtalk-20261120T030000Z.db   # 换成 data/backups/ 里要恢复的那一份（只写文件名）

  # 容器里运行机器人的账户，就是 compose.yaml 里 build.args 的 UID、GID（默认都是 1000）
  uid=$(docker compose run --rm --no-deps -T --entrypoint id bot -u)
  gid=$(docker compose run --rm --no-deps -T --entrypoint id bot -g)

  # 1. 先把备份复制成数据目录里的临时文件（同时确认它存在、可读、磁盘放得下），属主与 0600 一步设好；
  #    再用镜像里的 Python 只读打开它跑一遍 PRAGMA integrity_check，宿主机不用装 sqlite3
  staged=data/qtalk.db.restoring
  sudo install -m 600 -o "$uid" -g "$gid" "data/backups/$name" "$staged"
  docker compose run --rm --no-deps -T --entrypoint python bot -c '
import sqlite3, sys
conn = sqlite3.connect("file:/app/data/qtalk.db.restoring?mode=ro", uri=True)
result = conn.execute("PRAGMA integrity_check").fetchone()[0]
print("integrity_check:", result)
sys.exit(0 if result == "ok" else 1)
'

  # 2. 停机，把现在的库连同 -wal、-shm 一起挪进「恢复前」目录（已提交的数据可能还在 WAL 里，只复制主库会丢）
  docker compose stop bot
  keep="data/restore-$(date -u +%Y%m%dT%H%M%SZ)"
  sudo install -d -m 700 -o "$uid" -g "$gid" "$keep"
  for f in qtalk.db qtalk.db-wal qtalk.db-shm; do
    if sudo test -e "data/$f"; then sudo mv "data/$f" "$keep/"; fi
  done

  # 3. 放上校验过的备份，再启动
  sudo mv "$staged" data/qtalk.db
  docker compose start bot
  echo "已恢复；原来的库在 $keep"
)
```

数据目录属于容器账户，别的账户进不去，所以这里的文件操作都用 `sudo`。校验没通过时停在第 1 步，现在的库没动过；留下的 `data/qtalk.db.restoring` 可以删掉，换一份备份再试。

确认恢复无误后再删掉最后打印的那个 `data/restore-…` 目录。恢复错了想退回原来的库，同样整段执行（先把 `keep=` 换成那个目录）：先把恢复出来的那一套挪开，再把原来的挪回去。

```sh
(
  set -euo pipefail
  keep=data/restore-20261120T081500Z   # 换成恢复时最后打印的目录
  uid=$(docker compose run --rm --no-deps -T --entrypoint id bot -u)
  gid=$(docker compose run --rm --no-deps -T --entrypoint id bot -g)
  sudo test -e "$keep/qtalk.db"          # 原来的库确实在那里，才往下做
  docker compose stop bot
  wrong="data/restore-wrong-$(date -u +%Y%m%dT%H%M%SZ)"
  sudo install -d -m 700 -o "$uid" -g "$gid" "$wrong"
  for f in qtalk.db qtalk.db-wal qtalk.db-shm; do
    if sudo test -e "data/$f"; then sudo mv "data/$f" "$wrong/"; fi
  done
  for f in qtalk.db qtalk.db-wal qtalk.db-shm; do
    if sudo test -e "$keep/$f"; then sudo mv "$keep/$f" data/; fi
  done
  docker compose start bot
)
```

恢复的是备份那一刻的数据，之后的对话、授权与记忆都会丢失。

**异地备份**：本项目不加密备份。把 `data/backups/` 复制到别的机器或云存储之前，请自己加密（例如用 `age` 或 `gpg --symmetric`），**加密的密钥不要和备份放在一起**。只靠复制、不加密的异地备份等于把全部聊天记录交给了存储方。
