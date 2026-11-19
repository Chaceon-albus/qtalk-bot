"""按模型的输入能力筛选候选、决定要给用户的说明、组装带附件的上下文。

不碰网络、不依赖 pyqbot 与 SDK：读附件文件经 `MediaStore`（在线程里读），其余都是纯函数。

**能处理吗**（`accepts()`）：原件已存到本地，模型的 `inputs` 声明了这种输入，服务商接口接受这种实际的 MIME
类型（`llm.INPUT_LIMITS`），按 base64 计的大小也在它的内联上限之内。图片要 `image`，PDF 要 `file`，语音的 WAV
要 `audio`。

**放得进**（`fit()`）：一个模型按这一轮附件的顺序，把能接收的原件依次放进服务商的内联预算，放不下的跳过。
下面说的「能处理」都指放得进，而不是单件能接收。

**候选**（`plan_turn()`）：
1. 「必须照顾」的附件是这一轮里已存下的图片与文件（PDF），以及原声已存下、却没有识别文本的语音。每个模型
   数一数它能处理其中几个，取能处理得最多的那些模型；一个也处理不了时（包括这一轮没有这些附件）就是全部模型。
2. 有识别文本的语音不影响候选：照常由决策按内容选模型，选中能收音频的就给原声，收不了的就给识别文本。
3. 只留下实际能收到可答内容的模型（文字、放得进的原件、有识别文本的语音）；一个都没有时这一轮只发说明。
   所以没有识别文本的语音只有能听的模型答得了。
4. 候选按配置的顺序排列。只剩一个时不调用决策（见 `routing.Router.select()`）。

**备用模型**（`fallback_allowed()`）：选出的模型能处理的必须照顾的附件，备用模型都要能处理；备用模型实际能
收到的内容里也要有可答的。否则不换它。

**上下文**（`RequestBuilder`）：当前这一轮按选中模型的能力给原件或文字版；历史里最近
`[attachments] history_turns` 轮的图片在模型能处理、本地文件还在时给原件，其余一律文字版（音频和文件在历史里
只给文字版，免得每轮都重复花 token）。原件按「当前这一轮优先、历史从新到旧」占用服务商的内联上限，放不下的
退回文字版。连续的同一角色的消息合成一条（攒批后一轮有多条用户消息）。

历史里的文字版（`media.history_text()`）在原件当时确实交给过模型时（附件表的 `given`，回答那一轮记账时按实际
的请求记下）写清楚这一点、这里只保留描述、摘要或识别文本，免得模型以为自己从没看过、和它当时的回答自相矛盾；
没有这个记录的照旧用当时的文字版。本轮的原件与文字版不变。

只由表情组成的消息在取流水时已经跳过（见 `journal.py`），表情标记也从所有文字里去掉（`faces.clean()`）。

**分话题时的上下文**（里程碑 6a）：历史由几段合成（`merge_history()`）：不分话题的最近几轮，加上选中话题的最近
几轮，按消息去重、按对话里的位置排列，「往前数第几轮」按合成后的各轮重算，所以「最近几轮给原图」的规则对合成
出的各轮照常适用。话题目录与摘要经 `prompt.Memory` 放进系统提示词（`RequestBuilder.set_context()`）。
"""

import asyncio
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from . import texts
from .config import ModelSettings
from .faces import clean
from .journal import HistoryEntry, position
from .llm import (
    INPUT_LIMITS,
    AudioPart,
    ChatRequest,
    FilePart,
    ImagePart,
    Message,
    Modality,
    Part,
    Provider,
    Role,
    TextPart,
    inline_size,
)
from .media import (
    PDF,
    VOICE_UNRECOGNIZED,
    AttachmentKind,
    AttachmentRecord,
    AttachmentStatus,
    MediaStore,
    history_text,
    marker,
)
from .prompt import Memory, system_prompt

_MODALITIES = {
    AttachmentKind.IMAGE: Modality.IMAGE,
    AttachmentKind.VOICE: Modality.AUDIO,
    AttachmentKind.FILE: Modality.FILE,
}


def takes(model: ModelSettings, record: AttachmentRecord) -> bool:
    """`model` 收不收这种附件：原件已存到本地、`inputs` 声明了这种输入、服务商接受这种实际格式（不看大小）。"""
    modality = _MODALITIES.get(record.kind)
    return (
        modality is not None
        and record.status is AttachmentStatus.STORED
        and record.path is not None
        and record.mime_type is not None
        and modality in model.inputs
        and INPUT_LIMITS[model.provider].accepts(modality, record.mime_type)
    )


def accepts(model: ModelSettings, record: AttachmentRecord) -> bool:
    """`model` 能否直接接收这个附件的原件：收这种附件，大小（按 base64 计）也在服务商的内联上限之内。"""
    limit = INPUT_LIMITS[model.provider].max_inline_bytes
    return takes(model, record) and inline_size(record.size or 0) <= limit


def _required(current: Sequence[AttachmentRecord]) -> list[AttachmentRecord]:
    """必须照顾的附件：已存下的图片与 PDF，以及原声已存下、却没有识别文本的语音（收不了音频的模型只能看到
    「未识别出文字」，同批的文字不能替它作答）。有识别文本的语音不影响候选。"""
    return [
        r
        for r in current
        if r.status is AttachmentStatus.STORED
        and (
            r.kind in (AttachmentKind.IMAGE, AttachmentKind.FILE)
            or (r.kind is AttachmentKind.VOICE and r.text == VOICE_UNRECOGNIZED)
        )
    ]


def fit(model: ModelSettings, records: Sequence[AttachmentRecord]) -> frozenset[int]:
    """这一轮的附件里，按记录的大小、按顺序放得进 `model` 的内联预算的那些（附件编号）。

    逐个看，能接收、放得下就占用预算，放不下的跳过（后面更小的仍可能放得下）。候选评分与备用模型的检查以它
    为准；真正组装请求时 `RequestBuilder` 按同样的顺序、以实际读到的字节再确认一次。
    """
    budget = INPUT_LIMITS[model.provider].max_inline_bytes
    chosen: set[int] = set()
    for record in records:
        if not accepts(model, record):
            continue
        cost = inline_size(record.size or 0)
        if cost > budget:
            continue
        budget -= cost
        chosen.add(record.id)
    return frozenset(chosen)


def _answerable(
    current: Sequence[HistoryEntry], records: Sequence[AttachmentRecord], given: Collection[int]
) -> bool:
    """模型实际能收到的内容里有没有可以回答的：文字（去掉表情标记之后）、给了原件的附件，或有识别文本的语音。

    只由表情组成的消息（`faces_only`）在取流水时已经跳过，这里再守一道：它们不让一轮变得可答。
    """
    if any(clean(entry.content).strip() and not entry.faces_only for entry in current):
        return True
    return any(
        record.id in given
        or (record.kind is AttachmentKind.VOICE and record.text != VOICE_UNRECOGNIZED)
        for record in records
    )


def explain(
    records: Sequence[AttachmentRecord],
    *,
    models: Sequence[ModelSettings],
    viewers: Sequence[ModelSettings],
    given: Collection[int],
    missing: Collection[int] = (),
    max_size_mb: int,
) -> tuple[str, ...]:
    """要告诉用户的情况（去重，按出现的顺序）。

    `viewers` 是要交给的模型（最终的请求时就是选中的那一个），`given` 是实际给了原件的附件，`missing` 是记录
    还在、本地文件却读不到的附件，`models` 是配置里的全部模型（用来判断是不是没有谁收这种格式）。
    """
    notices: list[str] = []

    def notice(text: str) -> None:
        if text not in notices:
            notices.append(text)

    for record in records:
        match record.kind, record.status:
            case AttachmentKind.VOICE, _:
                if record.id not in given and record.text == VOICE_UNRECOGNIZED:
                    notice(texts.VOICE_UNRECOGNIZED)
            case _, AttachmentStatus.FAILED:
                notice(texts.ATTACHMENT_FAILED)
            case _, AttachmentStatus.TOO_LARGE:
                notice(texts.attachment_too_large(max_size_mb))
            case AttachmentKind.IMAGE, AttachmentStatus.UNSUPPORTED:
                notice(texts.IMAGE_FORMAT_UNSUPPORTED)
            case kind, _ if kind not in _MODALITIES:
                notice(texts.UNSUPPORTED)
            case _ if record.id in given:
                pass
            case _:
                notice(_why_not_given(record, models, viewers, missing))
    return tuple(notices)


def _why_not_given(
    record: AttachmentRecord,
    models: Sequence[ModelSettings],
    viewers: Sequence[ModelSettings],
    missing: Collection[int],
) -> str:
    """一张图片或一个文件没有给原件的原因。"""
    if record.id in missing:
        return texts.ATTACHMENT_GONE
    takers = [v for v in viewers if takes(v, record)]
    if takers:
        if not any(accepts(v, record) for v in takers):
            # 单件就超过了内联上限：说太大、要缩小，而不是格式不支持。
            limit = max(INPUT_LIMITS[v.provider].max_inline_bytes for v in takers) * 3 // 4
            return texts.attachment_too_large_for_model(limit // (1024 * 1024))
        return texts.ATTACHMENTS_OVER_BUDGET
    if record.kind is AttachmentKind.IMAGE:
        format_problem = (
            record.status is AttachmentStatus.STORED
            and any(Modality.IMAGE in m.inputs for m in models)
            and not any(takes(m, record) for m in models)
        )
        return texts.IMAGE_FORMAT_UNSUPPORTED if format_problem else texts.IMAGE_UNAVAILABLE
    if record.mime_type == PDF and record.status is AttachmentStatus.STORED:
        return texts.FILE_UNAVAILABLE
    return texts.FILE_UNSUPPORTED


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """一轮的筹划：候选模型（按配置的顺序）、有没有可以交给模型回答的内容，以及没有时要告诉用户的情况。

    有可答的内容时，给用户的说明要等选定模型、组装好请求后按实际的请求来定（`Built.notices`）。
    """

    candidates: tuple[str, ...]
    notices: tuple[str, ...]
    answerable: bool


def plan_turn(
    models: Sequence[ModelSettings],
    current: Sequence[HistoryEntry],
    *,
    max_size_mb: int,
    announce: Collection[int] | None = None,
) -> TurnPlan:
    """按这一轮的消息与附件筛选候选模型。规则见模块 docstring。

    `announce` 是要向用户说明情况的消息编号（这次新到的消息）；`None` 表示全部。并进这一轮的旧消息（之前
    已经说明过）只参加筛选。
    """
    records = [record for entry in current for record in entry.attachments]
    required = {r.id for r in _required(records)}
    fits = {m.name: fit(m, records) for m in models}
    scores = {m.name: len(fits[m.name] & required) for m in models}
    best = max(scores.values(), default=0)
    candidates = [m for m in models if scores[m.name] == best]
    # 语音不影响候选（由决策按内容选），但只留下实际收到的内容里有东西可答的模型（例如没有转写的纯语音，只有能听的模型答得了）。
    able = [m for m in candidates if _answerable(current, records, fits[m.name])]
    if able:
        candidates = able
    viewable = frozenset[int]().union(*(fits[m.name] for m in candidates))
    notices = explain(
        _announced(records, announce),
        models=models,
        viewers=candidates,
        given=viewable,
        max_size_mb=max_size_mb,
    )
    return TurnPlan(tuple(m.name for m in candidates), notices, bool(able))


def _announced(
    records: Sequence[AttachmentRecord], announce: Collection[int] | None
) -> list[AttachmentRecord]:
    return [r for r in records if announce is None or r.message_id in announce]


def fallback_allowed(
    fallback: ModelSettings, chosen: ModelSettings, current: Sequence[HistoryEntry]
) -> bool:
    """能不能换备用模型：选出的模型实际放得进的必须照顾的附件，备用模型都要放得进；备用模型实际能收到的
    内容里也要有可答的（没有转写的纯语音交给收不了音频的模型，它答不了还要计费）。"""
    records = [record for entry in current for record in entry.attachments]
    required = {r.id for r in _required(records)}
    mine, theirs = fit(chosen, records), fit(fallback, records)
    return (mine & required) <= theirs and _answerable(current, records, theirs)


def decision_text(entry: HistoryEntry) -> str:
    """决策题目里一条消息的文字：正文加上附件的简短标记，表情标记去掉（见 `_text()`）。"""
    pieces = [entry.content] if entry.content else []
    pieces.extend(marker(record) for record in entry.attachments)
    return clean("\n".join(pieces))


def decision_latest(current: Sequence[HistoryEntry], announce: Collection[int], limit: int) -> str:
    """决策题目里的 `latest`：这一轮的消息（含并进来的旧的待回复消息），总长不超过 `limit`。

    先给这次新到的消息（`announce`）留足额度，它们是触发这一轮的问题；剩下的额度再给旧消息，旧消息过长时
    保留靠后（更近）的部分，前面用省略号代替。
    """
    fresh = "\n".join(decision_text(e) for e in current if e.message_id in announce)
    older = "\n".join(decision_text(e) for e in current if e.message_id not in announce)
    if len(fresh) >= limit:
        return fresh[: limit - 1] + "…"
    room = limit - len(fresh) - 1  # 留一个换行
    if not older or room <= 1:
        return fresh
    if len(older) > room:
        older = "…" + older[len(older) - room + 1 :]
    return f"{older}\n{fresh}" if fresh else older


def decision_history(history: Sequence[HistoryEntry]) -> list[HistoryEntry]:
    return [HistoryEntry(entry.role, decision_text(entry)) for entry in history]


def merge_history(*groups: Sequence[HistoryEntry]) -> list[HistoryEntry]:
    """几段取自流水的历史合成一段：按消息编号去重，按对话里的位置排列（`journal.position()`：回复紧跟在它
    回答的消息之后），`turns_ago` 按合成后的各轮重算（每遇到一条更早的回复，就是往前的又一轮）。"""
    unique: dict[int, HistoryEntry] = {}
    loose: list[HistoryEntry] = []
    for group in groups:
        for entry in group:
            if entry.message_id is None:
                loose.append(entry)
            else:
                unique.setdefault(entry.message_id, entry)
    ordered = sorted([*loose, *unique.values()], key=position)
    merged: list[HistoryEntry] = []
    age = 0
    seen_user = False
    for entry in reversed(ordered):
        if entry.role is Role.ASSISTANT:
            if seen_user:
                age += 1
                seen_user = False
        else:
            seen_user = True
        merged.append(replace(entry, turns_ago=age))
    merged.reverse()
    return merged


@dataclass(frozen=True, slots=True)
class Built:
    """为某个模型组装好的请求。

    Attributes:
        request: 请求。
        answerable: 请求里有没有可以回答的内容；没有时不该调用模型。
        notices: 按这个请求实际给了什么，要告诉用户的情况（只含这次新到的消息的附件）。
        given: 这一轮的附件里实际以原件放进了请求的那些（附件编号）；历史里重新带上的图片不算。
    """

    request: ChatRequest
    answerable: bool
    notices: tuple[str, ...]
    given: frozenset[int] = frozenset()


class RequestBuilder:
    """为某个模型组装一轮的请求。同一轮里换备用模型时按它的能力重新组装。

    `current` 是这一轮的全部用户消息（含并进来的、之前没有回复的消息），`announce` 是其中这次新到的消息编号，
    只就它们向用户说明情况。
    """

    def __init__(
        self,
        store: MediaStore | None,
        *,
        user_id: int,
        persona: str,
        now: datetime,
        history: Sequence[HistoryEntry],
        current: Sequence[HistoryEntry],
        image_turns: int,
        max_output_tokens: int,
        models: Sequence[ModelSettings] = (),
        announce: Collection[int] | None = None,
        max_size_mb: int = 20,
        memory: Memory | None = None,
    ) -> None:
        self._store = store
        self._user_id = user_id
        self._persona = persona
        self._now = now
        self._history = history
        self._current = current
        self._image_turns = image_turns
        self._max_output_tokens = max_output_tokens
        self._models = models
        self._announce = announce
        self._max_size_mb = max_size_mb
        self._memory = memory

    def set_context(self, history: Sequence[HistoryEntry], memory: Memory | None) -> None:
        """换上选完话题之后才组装出的历史与记忆（分话题时，决策之后调用，组装请求之前）。"""
        self._history = history
        self._memory = memory

    def needs_originals(self, model: ModelSettings) -> bool:
        """组装给 `model` 的请求时会不会读附件原件（当前这一轮的，或最近几轮历史里的图片）。"""
        if self._store is None:
            return False
        return any(accepts(model, r) for r in self._current_records()) or any(
            accepts(model, r) for r in self._recent_images()
        )

    async def build(self, model: ModelSettings) -> Built:
        originals, missing = await self._originals(model)

        def in_current(record: AttachmentRecord) -> str:
            return record.text

        messages: list[Message] = []
        for entry, render in (
            *((e, history_text) for e in self._history),
            *((e, in_current) for e in self._current),
        ):
            if entry.faces_only:
                continue
            parts = _parts(entry, originals, render)
            if not parts:
                continue
            if messages and messages[-1].role is entry.role:
                previous = messages.pop()
                parts = _merge((*previous.parts, TextPart("\n"), *parts))
            messages.append(Message(entry.role, parts))
        request = ChatRequest(
            system=system_prompt(
                self._persona,
                self._now,
                process=model.show_process,
                preamble=model.provider is Provider.OPENAI and model.search,
                memory=self._memory,
            ),
            messages=tuple(messages),
            max_output_tokens=self._max_output_tokens,
        )
        records = self._current_records()
        notices = explain(
            _announced(records, self._announce),
            models=self._models or (model,),
            viewers=(model,),
            given=originals.keys(),
            missing=missing,
            max_size_mb=self._max_size_mb,
        )
        given = frozenset(r.id for r in records if r.id in originals)
        return Built(request, _answerable(self._current, records, originals.keys()), notices, given)

    def _current_records(self) -> list[AttachmentRecord]:
        return [r for entry in self._current for r in entry.attachments]

    def _recent_images(self) -> list[AttachmentRecord]:
        recent = [
            entry
            for entry in reversed(self._history)
            if entry.role is Role.USER and entry.turns_ago < self._image_turns
        ]
        return [r for entry in recent for r in entry.attachments if r.kind is AttachmentKind.IMAGE]

    async def _read(self, relative: str) -> bytes | None:
        """在线程里读一个原件。被取消时也等这次读完再让取消传出去：取消等待停不下已经开始的线程，提前返回会
        让调用方归还多模态名额，下一轮就开始读原件，内存里同时多出一份。读一个文件受大小约束，很快结束。"""
        assert self._store is not None
        reading = asyncio.ensure_future(
            asyncio.to_thread(self._store.read, self._user_id, relative)
        )
        try:
            return await asyncio.shield(reading)
        except asyncio.CancelledError:
            await asyncio.wait({reading})
            raise

    async def _originals(self, model: ModelSettings) -> tuple[dict[int, bytes], set[int]]:
        """要给原件的附件（编号 → 字节），以及记录还在、本地文件却读不到的当前附件。

        预算按实际读到的字节扣：本地文件与记录的大小不一致（如恢复了备份）时，以文件为准，放不下的不给。
        当前这一轮按顺序优先，历史里的图片从新到旧占用剩下的预算。
        """
        originals: dict[int, bytes] = {}
        missing: set[int] = set()
        if self._store is None:
            return originals, missing
        budget = INPUT_LIMITS[model.provider].max_inline_bytes
        for current, record in [
            *((True, r) for r in self._current_records()),
            *((False, r) for r in self._recent_images()),
        ]:
            if not accepts(model, record) or record.path is None:
                continue
            if inline_size(record.size or 0) > budget:
                continue  # 按记录就放不下，不必去读
            data = await self._read(record.path)
            if data is None:
                if current:
                    missing.add(record.id)
                continue
            cost = inline_size(len(data))
            if cost > budget:
                continue
            budget -= cost
            originals[record.id] = data
        return originals, missing


def _text(text: str) -> TextPart:
    """进入模型输入的一段文字。表情标记一律去掉：流水里保留实际收到、实际送达的原文，升级前的版本存过原始
    标记（用户消息、回显了标记的助手回复），附件的文字版（如文件名）里也可能有一个完整的标记。"""
    return TextPart(clean(text))


def _parts(
    entry: HistoryEntry,
    originals: dict[int, bytes],
    render: Callable[[AttachmentRecord], str],
) -> tuple[Part, ...]:
    """一条消息的各部分。`render` 给出不给原件时附件的文字版（历史与本轮的措辞不同）。"""
    content = clean(entry.content)
    parts: list[Part] = [TextPart(content)] if content.strip() else []
    for record in entry.attachments:
        data = originals.get(record.id)
        if data is None or record.mime_type is None:
            parts.append(_text(render(record)))
            continue
        match record.kind:
            case AttachmentKind.IMAGE:
                if record.quoted:
                    parts.append(TextPart("[被引用的图片：]"))
                parts.append(ImagePart(data, record.mime_type))
            case AttachmentKind.VOICE:
                parts.append(TextPart("[语音：]"))
                parts.append(AudioPart(data, record.mime_type))
            case AttachmentKind.FILE if record.mime_type == PDF:
                # 文字版里有文件名与大小，放在文件前面当作标签。
                parts.append(_text(record.text))
                parts.append(FilePart(data, record.mime_type))
            case _:
                parts.append(_text(record.text))
    return _merge(parts)


def _merge(parts: Sequence[Part]) -> tuple[Part, ...]:
    """相邻的文字合成一段（用换行连接）。"""
    merged: list[Part] = []
    for part in parts:
        if isinstance(part, TextPart) and merged and isinstance(merged[-1], TextPart):
            previous = merged[-1].text
            joiner = "" if previous.endswith("\n") or part.text.startswith("\n") else "\n"
            merged[-1] = TextPart(f"{previous}{joiner}{part.text}")
        else:
            merged.append(part)
    return tuple(merged)
