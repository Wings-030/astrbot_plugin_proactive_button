"""主动发言台。

在 AstrBot WebUI 的插件页面里选择会话与预设，向目标会话注入一条内部消息，
让 Bot 走 AstrBot 原生的事件管线生成并发送回复。

因此人格、对话历史、工具、消息防抖、分段回复、表情包插件、TTS 等行为
全部沿用 AstrBot 与其他插件的原有规则，本插件不自己调 LLM、也不自己发消息。
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from quart import request

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.platform.astrbot_message import MessageMember
from astrbot.core.platform.platform import PlatformStatus

PLUGIN_NAME = "astrbot_plugin_proactive_button"
API_PREFIX = f"/{PLUGIN_NAME}"
TASK_EXTRA_KEY = "proactive_button_task_id"

DEFAULT_PRESETS: list[dict[str, str]] = [
    {
        "id": "reply_latest",
        "name": "接话",
        "description": "顺着最近的聊天自然接一句",
        "prompt": (
            "阅读这个会话最近的聊天记录，顺着最后的话题自然地接一句，"
            "像普通群友一样。不要总结聊天记录，不要提到自己是被触发发言的，"
            "不要连续追问。"
        ),
    },
    {
        "id": "icebreak",
        "name": "群聊破冰",
        "description": "冷场时起一个大家能接的话题",
        "prompt": (
            "这个会话已经安静了一会儿。用符合人设的方式起一个轻松、"
            "大家都能接的话题，或者延续之前没聊完的内容。"
            "不要连续抛问题，不要像主持人。"
        ),
    },
    {
        "id": "roast",
        "name": "阴阳吐槽",
        "description": "针对最近的聊天阴阳一句",
        "prompt": (
            "根据最近的聊天内容，用阴阳怪气但不过分的方式吐槽一句，"
            "优先攻击离谱的逻辑和操作，不进行人身攻击，不骂脏话。"
        ),
    },
    {
        "id": "care",
        "name": "关心近况",
        "description": "问问对方在忙什么",
        "prompt": (
            "结合最近的聊天内容，自然地问问对方现在在做什么、进展怎么样。"
            "语气要像熟人，不要像客服回访，不要说教。"
        ),
    },
    {
        "id": "cute",
        "name": "撒娇卖萌",
        "description": "卖个萌或抱怨摸鱼",
        "prompt": (
            "用符合人设的方式撒个娇或卖个萌，可以抱怨摸鱼、算力、"
            "白米饭或加班。保持短句，不要刻意讨好。"
        ),
    },
    {
        "id": "night",
        "name": "晚安",
        "description": "道晚安或催睡觉",
        "prompt": (
            "结合当前时间，用符合人设的方式道晚安或催对方别熬夜。"
            "语气自然，不要长篇大论。"
        ),
    },
]

PRESET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MESSAGE_ID_PREFIX = "proactive-button-"


@register(
    PLUGIN_NAME,
    "Wings",
    "在 WebUI 里选会话和预设，立刻让 Bot 主动说一句（走原生回复管线）",
    "1.1.0",
)
class ProactiveButtonPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self.config = config or {}
        self.data_dir: Path = StarTools.get_data_dir(PLUGIN_NAME)
        self.presets_file = self.data_dir / "presets.json"

        self._presets: list[dict[str, str]] = []
        self._default_preset_id: str = ""
        self._load_presets()

        self._tasks: deque[dict[str, Any]] = deque(maxlen=40)
        self._task_index: dict[str, dict[str, Any]] = {}
        self._task_handles: dict[str, asyncio.Task] = {}
        self._running_sessions: set[str] = set()
        self._last_trigger_at: dict[str, float] = {}

        self._register_web_apis()

    # ------------------------------------------------------------------
    # 配置与预设
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            return default
        return default if value is None else value

    @staticmethod
    def _sanitize_presets(raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        result: list[dict[str, str]] = []
        used: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()[:40]
            prompt = str(item.get("prompt") or "").strip()
            if not name or not prompt:
                continue
            if len(prompt) > 4000:
                prompt = prompt[:4000]
            preset_id = str(item.get("id") or "").strip()
            if not PRESET_ID_RE.fullmatch(preset_id) or preset_id in used:
                preset_id = f"preset_{uuid.uuid4().hex[:8]}"
            used.add(preset_id)
            result.append(
                {
                    "id": preset_id,
                    "name": name,
                    "description": str(item.get("description") or "").strip()[:120],
                    "prompt": prompt,
                }
            )
            if len(result) >= 50:
                break
        return result

    def _load_presets(self) -> None:
        try:
            raw = json.loads(self.presets_file.read_text(encoding="utf-8"))
            presets = self._sanitize_presets(raw.get("presets"))
            if not presets:
                raise ValueError("presets 为空")
            self._presets = presets
            default_id = str(raw.get("default_id") or "").strip()
            self._default_preset_id = (
                default_id
                if any(p["id"] == default_id for p in presets)
                else presets[0]["id"]
            )
            return
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.logger.warning("[%s] 读取预设失败，使用内置默认值: %s", PLUGIN_NAME, exc)

        self._presets = [
            {**item, "description": item.get("description", "")}
            for item in DEFAULT_PRESETS
        ]
        self._default_preset_id = self._presets[0]["id"]
        try:
            self._save_presets()
        except Exception as exc:
            self.logger.warning("[%s] 写入默认预设失败: %s", PLUGIN_NAME, exc)

    def _save_presets(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "default_id": self._default_preset_id,
            "presets": self._presets,
        }
        tmp = self.presets_file.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.presets_file)

    # ------------------------------------------------------------------
    # Web API 注册
    # ------------------------------------------------------------------

    def _register_web_apis(self) -> None:
        register_api = getattr(self.context, "register_web_api", None)
        if register_api is None:
            self.logger.warning("[%s] 当前 AstrBot 不支持 register_web_api", PLUGIN_NAME)
            return
        register_api(f"{API_PREFIX}/sessions", self.api_sessions, ["GET"], "目标会话列表")
        register_api(f"{API_PREFIX}/presets", self.api_presets, ["GET"], "预设列表")
        register_api(
            f"{API_PREFIX}/presets/save", self.api_save_presets, ["POST"], "保存预设"
        )
        register_api(f"{API_PREFIX}/trigger", self.api_trigger, ["POST"], "触发主动发言")
        register_api(f"{API_PREFIX}/status", self.api_status, ["GET"], "触发任务状态")

    # ------------------------------------------------------------------
    # 会话枚举
    # ------------------------------------------------------------------

    def _sqlite_path(self) -> Path | None:
        try:
            url = str(getattr(self.context.get_db(), "DATABASE_URL", "") or "")
        except Exception:
            return None
        marker = ":///"
        index = url.find(marker)
        if index < 0:
            return None
        path = url[index + len(marker) :]
        return Path(path) if path else None

    @staticmethod
    def _to_umo(platform_id: str, user_id: str) -> str:
        uid = str(user_id or "").strip()
        if not uid:
            return ""
        if uid.count(":") >= 2:
            return uid
        if not platform_id:
            return ""
        message_type = (
            "GroupMessage"
            if ("@chatroom" in uid or uid.startswith("group_"))
            else "FriendMessage"
        )
        return f"{platform_id}:{message_type}:{uid}"

    @staticmethod
    def _describe_umo(umo: str) -> dict[str, str]:
        parts = umo.split(":", 2)
        if len(parts) != 3:
            return {"platform": "", "type": "未知", "target": umo, "label": umo}
        platform, message_type, target = parts
        if "Group" in message_type or "Guild" in message_type:
            type_label = "群聊"
        elif "Friend" in message_type or "Private" in message_type:
            type_label = "私聊"
        else:
            type_label = message_type or "未知"
        return {
            "platform": platform,
            "type": type_label,
            "target": target,
            "label": f"{type_label} {target}",
        }

    def _read_sessions_from_db(self) -> dict[str, str]:
        """从 AstrBot 数据库读取出现过的会话，返回 {umo: 最后活跃时间}。"""
        db_path = self._sqlite_path()
        if not db_path or not db_path.is_file():
            return {}
        sessions: dict[str, str] = {}
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        except Exception as exc:
            self.logger.warning("[%s] 打开数据库失败: %s", PLUGIN_NAME, exc)
            return {}
        try:
            cur = con.cursor()
            for sql in (
                "SELECT platform_id, user_id, MAX(created_at) "
                "FROM platform_message_history GROUP BY platform_id, user_id",
                "SELECT platform_id, user_id, MAX(updated_at) "
                "FROM conversations GROUP BY platform_id, user_id",
            ):
                try:
                    rows = cur.execute(sql).fetchall()
                except Exception:
                    continue
                for platform_id, user_id, last_at in rows:
                    platform_id = str(platform_id or "").strip()
                    if not platform_id or "webchat" in platform_id.lower():
                        continue
                    umo = self._to_umo(platform_id, str(user_id or ""))
                    if not umo:
                        continue
                    stamp = str(last_at or "")
                    if stamp > sessions.get(umo, ""):
                        sessions[umo] = stamp
        finally:
            con.close()
        return sessions

    def _active_platform_ids(self) -> set[str]:
        ids: set[str] = set()
        try:
            for platform in self.context.platform_manager.get_insts():
                platform_id = str(getattr(platform.meta(), "id", "") or "").strip()
                if platform_id:
                    ids.add(platform_id)
        except Exception as exc:
            self.logger.debug("[%s] 读取平台列表失败: %s", PLUGIN_NAME, exc)
        return ids

    async def _list_sessions(self) -> list[dict[str, str]]:
        sessions = await asyncio.to_thread(self._read_sessions_from_db)
        active_platforms = self._active_platform_ids()
        result: list[dict[str, str]] = []
        for umo, last_at in sessions.items():
            if active_platforms and umo.split(":", 1)[0] not in active_platforms:
                continue
            info = self._describe_umo(umo)
            result.append(
                {
                    "umo": umo,
                    "platform": info["platform"],
                    "type": info["type"],
                    "target": info["target"],
                    "label": info["label"],
                    "last_active": last_at,
                }
            )
        result.sort(key=lambda item: item.get("last_active", ""), reverse=True)
        return result[:60]

    # ------------------------------------------------------------------
    # 原生事件注入
    # ------------------------------------------------------------------

    @staticmethod
    def _build_event_prompt(base_prompt: str, extra: str) -> str:
        lines = [
            "[主动发言请求]",
            "请你现在主动在这个会话里说一句话，不要等我先开口。",
            "",
            f"本次要求：{base_prompt.strip()}",
        ]
        if extra.strip():
            lines.extend(["", f"补充要求：{extra.strip()}"])
        lines.extend(
            [
                "",
                "请直接输出要发到聊天里的内容。不要解释、不要复述这段请求，"
                "不要提到“主动发言”“系统任务”之类的内部信息。",
            ]
        )
        return "\n".join(lines)

    def _resolve_sender(self, umo: str, target_id: str, is_group: bool) -> tuple[str, str]:
        if not is_group:
            return target_id, "Master"
        try:
            admins = self.context.get_config(umo).get("admins_id", []) or []
        except Exception:
            admins = []
        for admin_id in admins:
            admin_id = str(admin_id or "").strip()
            if admin_id:
                return admin_id, "Master"
        return "proactive_button", "系统"

    def _find_adapter(self, platform_id: str):
        instances = self.context.platform_manager.get_insts()
        adapter = next(
            (item for item in instances if item.meta().id == platform_id), None
        )
        if adapter is None:
            adapter = next(
                (item for item in instances if item.meta().name == platform_id), None
            )
        return adapter

    async def _inject_trigger_event(self, umo: str, prompt: str, task_id: str) -> None:
        platform_id, message_type, target_id = umo.split(":", 2)
        is_group = "Group" in message_type or "Guild" in message_type

        adapter = self._find_adapter(platform_id)
        if adapter is None:
            raise RuntimeError(f"找不到平台适配器: {platform_id}")
        if getattr(adapter, "status", None) != PlatformStatus.RUNNING:
            raise RuntimeError(f"平台未在运行: {platform_id}")

        self_id = str(getattr(adapter, "self_id", "") or platform_id)
        sender_id, sender_name = self._resolve_sender(umo, target_id, is_group)
        message = await StarTools.create_message(
            type="GroupMessage" if is_group else "FriendMessage",
            self_id=self_id,
            session_id=target_id,
            sender=MessageMember(user_id=sender_id, nickname=sender_name),
            message=[Plain(text=prompt)],
            message_str=prompt,
            message_id=f"{MESSAGE_ID_PREFIX}{task_id}",
            group_id=target_id if is_group else "",
        )
        event = adapter.create_event(message)
        event.is_wake = True
        event.is_at_or_wake_command = True
        event.set_extra(TASK_EXTRA_KEY, task_id)
        adapter.commit_event(event)

    # ------------------------------------------------------------------
    # 原生管线钩子：只处理本插件注入的事件
    # ------------------------------------------------------------------

    @staticmethod
    def _task_id_from_event(event: AstrMessageEvent) -> str:
        task_id = str(event.get_extra(TASK_EXTRA_KEY) or "").strip()
        if task_id:
            return task_id
        # 防抖插件重投事件时会重建事件且不复制 extra，但会保留 message_id。
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        if message_id.startswith(MESSAGE_ID_PREFIX):
            return message_id[len(MESSAGE_ID_PREFIX) :]
        return ""

    @filter.on_llm_response()
    async def _capture_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        task_id = self._task_id_from_event(event)
        if not task_id:
            return
        record = self._task_index.get(task_id)
        if not record:
            return
        record["text"] = str(resp.completion_text or "").strip()

    @filter.after_message_sent()
    async def _mark_sent(self, event: AstrMessageEvent):
        task_id = self._task_id_from_event(event)
        if not task_id:
            return
        record = self._task_index.get(task_id)
        if not record or record.get("state") != "running":
            return
        record.update({"state": "ok", "finished_at": time.time(), "error": ""})
        self._finish_task(task_id)

    # ------------------------------------------------------------------
    # 任务状态
    # ------------------------------------------------------------------

    def _finish_task(self, task_id: str) -> None:
        record = self._task_index.pop(task_id, None)
        if record:
            self._running_sessions.discard(record.get("umo", ""))
        handle = self._task_handles.pop(task_id, None)
        if handle and not handle.done():
            handle.cancel()

    async def _watch_task(self, task_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
            record = self._task_index.get(task_id)
            if record and record.get("state") == "running":
                record.update(
                    {
                        "state": "error",
                        "error": f"等待 {int(timeout)} 秒仍未发送，可能被其他插件拦截或模型没有回复",
                        "finished_at": time.time(),
                    }
                )
                self._finish_task(task_id)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # API 处理
    # ------------------------------------------------------------------

    async def api_sessions(self):
        return {"ok": True, "sessions": await self._list_sessions()}

    async def api_presets(self):
        return {
            "ok": True,
            "presets": self._presets,
            "default_id": self._default_preset_id,
        }

    async def api_save_presets(self):
        payload = await request.get_json(silent=True) or {}
        presets = self._sanitize_presets(payload.get("presets"))
        if not presets:
            return {"ok": False, "error": "至少保留一个有效的预设"}, 400

        default_id = str(payload.get("default_id") or "").strip()
        if not any(item["id"] == default_id for item in presets):
            default_id = presets[0]["id"]

        self._presets = presets
        self._default_preset_id = default_id
        try:
            await asyncio.to_thread(self._save_presets)
        except Exception as exc:
            self.logger.error("[%s] 保存预设失败: %s", PLUGIN_NAME, exc)
            return {"ok": False, "error": f"保存失败: {exc}"}, 500
        return {
            "ok": True,
            "presets": self._presets,
            "default_id": self._default_preset_id,
        }

    async def api_trigger(self):
        if not bool(self._cfg("enabled", True)):
            return {"ok": False, "error": "插件已在配置中关闭"}, 403

        payload = await request.get_json(silent=True) or {}
        umo = str(payload.get("umo") or "").strip()
        if len(umo.split(":")) < 3:
            return {"ok": False, "error": "会话 UMO 不合法"}, 400

        preset_id = str(payload.get("preset_id") or "").strip()
        base_prompt = ""
        preset_name = "自定义"
        if preset_id == "custom" or not preset_id:
            base_prompt = str(payload.get("prompt") or "").strip()
        else:
            preset = next(
                (item for item in self._presets if item["id"] == preset_id), None
            )
            if not preset:
                return {"ok": False, "error": "找不到该预设"}, 404
            base_prompt = preset["prompt"]
            preset_name = preset["name"]

        if not base_prompt:
            return {"ok": False, "error": "提示词不能为空"}, 400
        if len(base_prompt) > 4000:
            return {"ok": False, "error": "提示词过长（最多 4000 字）"}, 400

        extra = str(payload.get("extra") or "").strip()[:2000]

        if umo in self._running_sessions:
            return {"ok": False, "error": "该会话正在触发中，请稍候"}, 409

        cooldown = int(self._cfg("cooldown_seconds", 5) or 0)
        now = time.time()
        if cooldown > 0 and now - self._last_trigger_at.get(umo, 0) < cooldown:
            return {"ok": False, "error": f"冷却中，请 {cooldown} 秒后再试"}, 429

        info = self._describe_umo(umo)
        task_id = uuid.uuid4().hex[:12]
        record: dict[str, Any] = {
            "id": task_id,
            "umo": umo,
            "session_label": info["label"],
            "session_type": info["type"],
            "preset_id": preset_id,
            "preset_name": preset_name,
            "state": "running",
            "text": "",
            "error": "",
            "started_at": now,
            "finished_at": None,
        }

        self._tasks.appendleft(record)
        self._task_index[task_id] = record
        self._running_sessions.add(umo)
        self._last_trigger_at[umo] = now

        timeout = float(self._cfg("response_timeout_seconds", 180) or 180)
        self._task_handles[task_id] = asyncio.create_task(
            self._watch_task(task_id, timeout)
        )

        prompt = self._build_event_prompt(base_prompt, extra)
        try:
            await self._inject_trigger_event(umo, prompt, task_id)
        except Exception as exc:
            self.logger.error("[%s] 注入事件失败: %s", PLUGIN_NAME, exc, exc_info=True)
            record.update(
                {"state": "error", "error": str(exc), "finished_at": time.time()}
            )
            self._finish_task(task_id)
            return {"ok": False, "error": str(exc)}, 500

        return {"ok": True, "task_id": task_id, "task": dict(record)}

    async def api_status(self):
        return {
            "ok": True,
            "tasks": list(self._tasks),
            "running_sessions": sorted(self._running_sessions),
        }

    async def terminate(self):
        for handle in list(self._task_handles.values()):
            if not handle.done():
                handle.cancel()
        self._task_handles.clear()
        self._running_sessions.clear()
        self._task_index.clear()
