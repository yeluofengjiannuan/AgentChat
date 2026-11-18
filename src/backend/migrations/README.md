# 数据库迁移（Alembic）

表结构不再由 `create_all` 决定，改由 Alembic 版本管理。后端启动时会自动执行 `alembic upgrade head`。

## 目录

```
src/backend/
├── alembic.ini
└── migrations/
    ├── env.py                # 从 AGENTCHAT_CONFIG 解析数据库 URL，注册全部 SQLModel 表
    ├── script.py.mako        # 新迁移模板（已 import sqlmodel，AutoString 类型可用）
    └── versions/
        └── b0350cc48709_baseline_existing_schema.py   # 基线：现有 22 张表
```

## 已有数据库

当前 `agentchat` 库已经 `alembic stamp head`，记为基线版本，不会重复建表。

## 日常改表流程

1. 修改 `agentchat/database/models/*.py`；
2. 生成迁移：

```powershell
cd src\backend
$env:AGENTCHAT_CONFIG="agentchat/config.local.simple.yaml"
.\.venv\Scripts\alembic.exe revision --autogenerate -m "add xxx column"
```

3. 检查 `migrations/versions/` 下新生成的文件（Alembic 对列重命名/默认值偶尔识别不准，需要人工确认）；
4. 应用：

```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

也可以直接重启后端，启动时会自动 upgrade。

## 新环境 / 空库

```powershell
# 建库后直接 upgrade，会自动建出全部表
.\.venv\Scripts\alembic.exe upgrade head
```

## 生成基线迁移（仅一次，已执行过）

用空的临时库对比模型生成 DDL，再把真实库 stamp 到该版本：

```powershell
$env:AGENTCHAT_MIGRATION_URL="mysql+pymysql://root:密码@127.0.0.1:3306/agentchat_baseline_tmp"
.\.venv\Scripts\alembic.exe revision --autogenerate -m "baseline: existing schema"
.\.venv\Scripts\alembic.exe upgrade head                     # 在临时库验证
.\.venv\Scripts\alembic.exe stamp head                       # 真实库标记（不执行 DDL）
```
