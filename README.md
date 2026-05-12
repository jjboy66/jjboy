# VisionGuard-Lite

VisionGuard-Lite 是一个面向中小场景的目标检测系统示例项目，提供：

- 训练框架（含配置化训练/评估脚本）
- 推理服务（FastAPI）
- 告警规则引擎（区域入侵、停留超时、危险目标）
- 主动学习闭环接口（误报反馈、重训任务请求）
- 可视化看板（基础 dashboard）

## 目录结构

```text
configs/              # 模型、数据、训练、部署配置
data/                 # 数据目录（raw/annotations/splits/processed）
src/
  api/                # 推理与告警接口
  datasets/           # 数据清单读取与写入
  infer/              # 推理流水线与流输入抽象
  models/             # 检测器接口与默认实现
  postprocess/        # NMS/WBF/时序平滑/规则引擎
  train/              # 训练与评估逻辑
  ui/                 # 前端看板
tools/                # 数据转换、标注质检、训练、评估、导出
tests/                # 单元与接口测试
```

## 快速开始

### 1) 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2) 训练与评估

```bash
python tools/train.py
python tools/evaluate.py
```

### 3) 导出模型占位物

```bash
python tools/export_model.py
```

### 4) 启动 API

```bash
python tools/run_api.py
```

接口示例：
- `GET /health`
- `POST /detect`
- `GET /alerts`
- `POST /feedback`
- `POST /active-learning/retrain-request`

### 5) 运行测试

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 主动学习闭环说明

- 在线推理后，低置信度与误报样本通过 `/feedback` 回流至 `data/annotations/feedback.jsonl`
- 可通过 `/active-learning/retrain-request` 触发周期增量训练任务入队
- 训练脚本可按 manifest 规模输出基线指标（mAP50-95/Recall/Small AP）用于迭代对比

## 部署建议

- 云端：FastAPI + Uvicorn + PostgreSQL + 对象存储
- 边端：导出 ONNX/TensorRT，结合量化与蒸馏做实时部署
- 监控：延迟/FPS/资源占用/误报漏报率
