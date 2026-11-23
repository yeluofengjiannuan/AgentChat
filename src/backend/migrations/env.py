"""Alembic 迁移环境。

数据库 URL 的优先级：
1. 启动时由 init_data.run_migrations() 传入的 sqlalchemy.url；
2. 环境变量 AGENTCHAT_MIGRATION_URL（生成基线迁移、临时库校验用）；
3. 配置文件 AGENTCHAT_CONFIG（默认依次尝试 config.local.simple.yaml / config.local.yaml / config.yaml）。
"""
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _config_path() -> Path:
    selected = os.getenv("AGENTCHAT_CONFIG")
    if selected:
        path = Path(selected)
        return path if path.is_absolute() else BACKEND_DIR / path
    for name in ("config.local.simple.yaml", "config.local.yaml", "config.yaml"):
        candidate = BACKEND_DIR / "agentchat" / name
        if candidate.is_file():
            return candidate
    return BACKEND_DIR / "agentchat" / "config.yaml"


def _load_app_settings() -> Path:
    """导入 agentchat.database 前必须先把配置注入共享 app_settings（模块导入期会读它）。"""
    from agentchat.settings import app_settings, load_settings

    path = _config_path()
    settings = load_settings(path)
    for key in settings.__fields__:
        setattr(app_settings, key, getattr(settings, key))
    return path


def _database_url() -> str:
    # 无论是否覆盖 URL，都要先注入配置：导入 agentchat.database 时会读取 app_settings。
    _load_app_settings()
    override = os.getenv("AGENTCHAT_MIGRATION_URL") or config.get_main_option("sqlalchemy.url")
    if override and override.strip():
        return override.strip()

    from agentchat.database.connection import resolve_database_urls
    from agentchat.settings import app_settings

    urls = resolve_database_urls(app_settings.database or app_settings.mysql)
    return urls.sync.render_as_string(hide_password=False)


config.set_main_option("sqlalchemy.url", _database_url())

# 导入 database 包会注册全部 SQLModel 表（metadata 的完整来源）。
from sqlmodel import SQLModel  # noqa: E402

import agentchat.database  # noqa: E402,F401

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    """离线模式：只输出 SQL，不连数据库。"""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连数据库执行迁移。"""
    from sqlalchemy import engine_from_config, pool

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
