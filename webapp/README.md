# 地图海陆重建 · 网页测评版

这是主文件夹批量脚本（`run_map_experiments.py`）的网页对应版本，题目、参照矩阵和评分口径完全一致。
全部代码放在 `webapp/` 目录内，**不读写主目录的其他文件**，与主文件夹保持隔离：

- 它不读取 `experiments/`、`quant/submissions/` 或任何既有结果；
- 它不写入主目录，只在浏览器内存里保存本批结果，需要时手动下载；
- 任务定义、题目、参照矩阵都是从主目录资源**复制/生成**出来的静态副本。

## 1. 是否需要后端

**默认不需要后端。** 打开 `index.html` 即可用，所有请求都在 JS 里发出，所有评分也在浏览器里完成。

只有在供应商接口不允许浏览器跨域（CORS）时，才需要可选的 Flask 中继：

| 模式 | 说明 |
| --- | --- |
| 浏览器直连（默认） | 页面直接 POST 到接口；要求接口返回 CORS 头 |
| 本地 Flask 代理 | 浏览器只跟 `127.0.0.1:8788` 通信，由它转发到接口；Key 不落盘 |

## 2. 快速开始（无后端）

1. 双击 `webapp/index.html`，或把整个 `webapp/` 放到任意静态服务器上；
2. 填写接口地址、API Key、模型 ID；
3. 选择「接口类型」（Chat Completions / Responses API）与「请求方式」（浏览器直连）；
4. 勾选要测的地图，设置每组重复次数，点「开始测试」；
5. 结果表逐行出现，点任意一行可在预览区查看判定图与误差图。

> `file://` 直接打开时页面自身脚本可正常运行；如果接口没有 CORS 头，浏览器会拦住跨域请求，这时改用下面的 Flask 代理。

## 3. 可选后端（Flask）

```powershell
pip install flask
python webapp/server.py                 # 默认 http://127.0.0.1:8788/
python webapp/server.py --port 9000 --no-browser
```

启动后：

- 浏览器打开 `http://127.0.0.1:8788/`；
- 在页面上把「请求方式」改成「本地 Flask 代理」；
- 代理地址默认 `http://127.0.0.1:8788/api/proxy`。

代理只做转发和 CORS 头处理，不记录、不保存 API Key，也不写任何日志文件。

## 4. 任务与评分口径

| 地图 | 尺寸 | 采样点 | 归一化 |
| --- | --- | --- | --- |
| 全球 2° | 90×180 | 16,200 | 16,200 |
| 南海 0.2° | 111×141 | 15,651 | 16,200 |
| 地中海 0.2° | 71×231 | 16,401 | 16,200 |
| 濑户内海 2/63° | 64×253 | 16,192 | 16,200 |
| 北极航道 0.1° | 61×241 | 14,701 | 16,200 |
| 赫卡特海峡 2/126° | 127×127 | 16,129 | 16,200 |

与主脚本一致：

- `得分 = 采样点数 − 与参照矩阵不一致的格子数`；
- 区域图同时给出归一化到 16,200 的分数；
- PNG 解码：先转灰度（ITU-R 601-2），再以 `127.5` 为阈值二值化；
- 尺寸处理：原尺寸直接用；整数倍放大按块多数表决；其他尺寸用 BOX 重采样并标记 `lossy`；
- 结果表另外给出准确率、HSS、边界 F1@1、虚报/漏报格数，以及“可能翻转”提示。

题目会自动追加一段输出格式说明（`BEGIN_MATRIX` / `END_MATRIX`、每行连续 0/1）。
因为浏览器不能替模型执行 Python，网页版只要求模型**直接输出判断矩阵**，然后由页面解析评分；
主脚本的“让模型写脚本再代跑”流程仍然以前者为准。

## 5. 部署到静态服务器

`webapp/` 是纯静态目录，除 `server.py` 外的文件都可以直接托管：

```text
webapp/
  index.html
  app.css
  js/          页面脚本（含生成的 assets-data.js）
  server.py    可选 Flask 中继
  tools/       资源生成脚本
  tests/       自测
```

若接口允许 CORS，把 `webapp/` 丢到任意静态托管即可，无需后端。

## 6. 维护与自测

```powershell
# 题目或参照矩阵更新后，重新生成内嵌资源
node webapp/tools/build_assets.js

# 评分核心单元测试（六个形状、满分自检、格式往返、翻转折扣分）
node webapp/tests/score.test.js

# 可选 Flask 中继的单元测试（用桩模块，无需安装 Flask）
python webapp/tests/server.test.py
```

`webapp/__pycache__/` 由 Python 自动生成，已在 `webapp/.gitignore` 中排除。

端到端冒烟测试需要先启动静态服务器，再启动一个带调试端口的无头 Chromium 系浏览器（Edge 或 Chrome 均可）：

```powershell
# 1) 静态服务器
python -m http.server 8765 --bind 127.0.0.1 --directory webapp

# 2) 无头浏览器（另开一个窗口；换成你本机 msedge.exe / chrome.exe 的路径）
$browser = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
& $browser --headless=new --disable-gpu --no-first-run `
  --remote-debugging-port=9333 --user-data-dir="$env:TEMP\edge-cdp" about:blank

# 3) 运行冒烟测试（内置一个假的 OpenAI 兼容接口）
node webapp/tests/smoke.e2e.js
```

端口与入口地址可以用环境变量覆盖：`SMOKE_CDP`（默认 `http://127.0.0.1:9333`）、`SMOKE_PAGE`（默认 `http://127.0.0.1:8765/`）。

冒烟测试覆盖：页面启动、接口请求、矩阵解析、评分、手动文本评分、PNG 8× 放大解码，以及无控制台报错。
用环境变量可以切换被测路径与地图：

```powershell
$env:SMOKE_TASK='global'        # global | csc | dzh | seto | arctic01 | hecate
$env:SMOKE_TRANSPORT='proxy'    # direct | proxy
$env:SMOKE_WIRE='responses'     # chat | responses
$env:SMOKE_STREAM='1'           # 1 走 SSE 流式分支
$env:SMOKE_PAGE='file:///.../webapp/index.html'   # 默认 http://127.0.0.1:8765/
node webapp/tests/smoke.e2e.js
Remove-Item Env:\SMOKE_TASK,Env:\SMOKE_TRANSPORT,Env:\SMOKE_WIRE,Env:\SMOKE_STREAM,Env:\SMOKE_PAGE
```

已验证组合：`chat` 与 `responses` 两种接口、各自 JSON 与 SSE 流式、`direct` 与 `proxy` 两种传输，共 5 条路径全部通过；
六张地图（global / csc / dzh / seto / arctic01 / hecate）分别跑通满分自检；
`http://` 静态服务与 `file://` 直接打开两种方式也都验证通过。测试还会覆盖“截断回复被报为解析失败且原始回复可下载”。

## 7. 安全提示

- API Key 只保存在你自己的浏览器里（勾选“记住 Key”才写入 `localStorage`）；不勾选则仅存在于当前页面内存。
- 页面不会把 Key 写进任何下载文件；下载的 JSON 只包含任务、参数和评分。
- Flask 中继把你的 Key 透露给浏览器所连的接口，但不会记录到磁盘。