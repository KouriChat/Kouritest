# 运行中修订记录

时间：2026-10-28 16:47（Asia/Shanghai）

首波中 channel-A / `gpt-6.1-sol-codex` 的两个并发请求均长时间停留在流式响应阶段，未生成可校验交付；其中一条在 16:44:54 记录 `stream ended before response.completed`，随后进入重试。为避免单 provider 阻塞全部 6 个波次，终止该挂起运行并从后续平衡设计中排除 channel-A。

- 未改动：任务、提示词、工具、输出上限、评分器、预算水平、重复数。
- 修订后计划：channel-C 与channel-F × 2 tasks × 3 levels × 2 replicates = 24 个模型运行。
- 修订前计划仍保存在 `schedule_initial_3providers.json`，共 36 个。
- channel-A 的运行目录、流日志和重试日志保留为 provider 级失败证据，不纳入有效质量样本。
- 首波 channel-C/channel-F 在 `medium` 下的 `seto` 均完成矩阵与图像生成，但方向校验失败；按协议保留为 partial，不人工翻转。
