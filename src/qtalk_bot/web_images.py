"""回答里的网络图片：模型写的 `![说明](地址)` 由机器人下载、传到素材库、在正文里改写成带宽高的素材地址
（做法见 `docs/design/presentation.md`「图片随回答发出」）。这里只有不联网的部分：

- **白名单**（`DEFAULT_HOSTS`、`image_hosts()`、`url_allowed()`）：只下载主机名落在白名单里的地址。匹配复用附件
  下载的规则（`media.known_host()`）：主机名要是合法的域名（IP 字面量不算），等于某个后缀或以「.后缀」结尾，
  按整段比较，所以单段的 `gov` 不会命中 `data.gov.au`、`evilgov.cn`。下载时每一跳仍由下载器按同一份名单检查。
- **下载计划**（`download_plan()`）：先按白名单检查模型写的地址，通过了才给出依次尝试的地址。`upload.wikimedia.org`
  的原图地址会被限流，缩略图不会：先按宽度从大到小试几个缩略图地址，最后才是原地址。
- **读尺寸**（`image_size()`）与**显示尺寸**（`display_size()`）：只看文件头；平台的图片语法要写宽高，并按原图的
  比例。
- **流式扫描**（`ImageScanner`）：从模型的输出流里认出图片语法，其余文字原样交出。
- **改写**（`render_image()`、`fallback_text()`）：素材库地址的图片语法，以及去掉图片时留下的说明文字。

扫描器放在 `MarkdownGuard` 之后（它内部带着一个 guard，先过 guard 再扫描），理由：

1. 改写出来的图片语法引用 `qqbot-media://` 这类非 http 地址；放在 guard 之前的话，guard 会把它的圆括号换成全角，
   图片就坏了。放在之后，改写结果直接交给发送端，不再经过 guard。
2. guard 的分段带着对应的原文（`Segment.source`，与输出等长、一一对应），扫描器在原文上认图片，`ImageRef` 的
   `alt`、`url`、`raw` 都取自原文：地址里有 guard 会改的括号（如 `a](b).png`）也不会被改掉。不是图片的部分照旧
   输出 guard 改过的文字。
3. guard 的状态（围栏、行内代码、行首）按模型的原文推进，之后把图片换成别的文字不影响它；`guard.closing_fence()`
   照常可用。

扫描的规则：

- 认 `![说明](地址)`、`![说明](<地址>)`，地址后面可以有用空白隔开、引号或圆括号括起的标题；说明里可以有成对的
  方括号与反斜杠转义。整段语法里不能有没转义的 `|`（表格会在那里拆开单元格）。地址按 CommonMark 处理反斜杠转义（反斜杠加 ASCII 标点是这个字符本身：被转义的圆括号不参与
  配平，`\\>` 不结束尖括号地址）与以分号结尾的字符引用；地址完整之后先解码，再判断协议：解码之后以 `http://`、
  `https://` 开头（不分大小写）、字符引用没有解码出空白与控制字符的才算图片（`ImageRef.url` 是解码之后的）。尖括号
  地址里字面的空格写成 `%20`，别的空白与控制字符不算图片。
  说明文字同样解码转义与字符引用。
- 每个字符分成三类：**正文**（认图片）、**代码**（不认图片，但 `![` 照样换成全角，见下）、**原样放行**（一个字也
  不改）。由两套互不调用、各自直接看原文的机制决定，分工如下：
  - **原样放行要三方都同意**：闸门 `_Gate` 说可以（到目前为止的全文是它认的「规范形状」，这里是开头行从第 0 列
    开始、恰好 3 个反引号的围栏的内容，结束行同样是恰好 3 个；用正则成对匹配反引号的解析器看也是同样的范围），
    块跟踪器 `_BlockTracker` 说是顶层、开头行从第 0 列开始的反引号围栏代码块的内容，guard 也把这些字符当作反引号
    围栏（分条时靠 guard 补上与重开围栏）。少一个就按代码处理。另外还有字节上限（见下）；`start_bytes` 不为零
    （扫描器的输出前面还有别的内容）时整条回答都不放行。下面那条「输出里没有 ASCII 的 `![`」的保证（安全）只靠
    这一条成立。闸门只看模型的原文，看不到我们换上去的显示文字，所以改写的图片语法与兜底文字（`clean_alt()`）
    里不能有闸门拿不准的任何标记。
  - **其余位置是正文还是代码只看 `_BlockTracker`**：它按 CommonMark（以参照的 markdown-it 为准）逐个字符地跟踪
    块引用与列表项、任何一层里的围栏代码块与缩进代码块、段落里的行内代码、链接引用定义，决定的是认得准不准：
    不把代码里的写法当成图片（不多认），正常写法的图片不漏掉（不少认）。拿不准是不是代码的按代码算，只会少认。
  - 为什么两个都要：`_BlockTracker` 精细，但精细的模拟只要错一处（例如把分隔线 `- - -` 当成了列表项、于是把顶层
    围栏的内容当成新围栏的开头），单靠它与 guard 决定放行，模型的 `![` 就会原样漏出去，而这类输入随机测试未必碰得到。
    原样放行还假定 QQ 认代码块的规则与 CommonMark 一致，而 QQ 另有块公式（`$$`、`\\[…\\]`），各家对缩进的围栏、
    波浪线围栏、行中间的反引号串、单独的 `\\r`、front matter、Unicode 空白与不可见字符看法也不一。`_Gate` 不模拟哪一家，只认一种简单的形状，一见到这些
    拿不准的写法就从此不再放行（规则见它的 docstring），它错只会错在少放行；但它分不出列表、引用、行内代码，单靠它
    认图片会多认、少认。所以安全交给粗而保守的闸门，认得准交给精细的块跟踪器，`_BlockTracker` 判断错了只影响哪里
    认图片，不会让 `![` 原样漏出去。
- 原样放行的字节上限：平台把超过约 15000 字节的流式消息不看结构地硬切成几条气泡，切口落在围栏里时，后一条从围栏
  中间开始，里面的 `![` 就成了正文。所以扫描器记下自己输出了多少字节（文字按 UTF-8，每张图按 `IMAGE_DISPLAY_BYTES`
  算，它显示时最多占这么多），一处 ASCII 的 `![` 只有在「`start_bytes` + 它之前输出的字节数 + 2 ≤
  `VERBATIM_LIMIT_BYTES`」时才原样留着，否则感叹号照样换掉。按每一处 `![` 的确切位置判断，不随切分变化。
- 链接的 `](地址 "标题")` 部分（语法同图片的后半段，宽一些：任何协议、可以跨行）里的图片写法只是地址或标题的
  文字，不认；链接的文字（方括号里）照常认。只有 `]` 前面有可配的 `[` 时才这样看：数正文里没被转义、还没配上的
  `[`（遇到空行清零）；开头的 `[` 被转义了、或根本没有时，`](` 只是字符，括号里照常认图片。这一部分被代码截断时拿不准它在哪里结束，从 `](` 算起
  `MAX_SYNTAX_CHARS` 个字符里（截断它的代码也算在内）的图片写法都不认，遇到空行为止（链接的地址与标题都不能跨过
  空行）；这些字符的最后一个是起转义作用的反斜杠时，被它转义的那个字符也算在内。
- 没认成图片的 `![` 的感叹号都换成全角的「！」（`DEFUSED_BANG`），不论原因：写法不对、超过长度上限、中间有换行、
  地址不是 http(s)、在代码或链接的地址里、被代码截断，或到 `finish()` 时还没写完。反斜杠转义的 `\\!` 不是图片的
  开头，但后面紧跟 `[` 时同样换成全角（平台认不认这个转义没有实测）。后面紧跟不可见字符（`_invisible()`，如零宽
  空格）的 `!` 也换成全角：有的解析方式忽略这些字符，`!` + 零宽空格 + `[` 按那种方式仍是 `![`；只看后面一个字符，
  不可见字符原样留着（按 CommonMark 它本来就不是图片，不影响认图片）。原样放行的内容里不用这样处理：闸门找标记时
  已经跳过不可见字符，按那种解析方式看它们仍在代码块里。其余的 `!` 不动。
- 由此得到一条简单、可检验的保证：除了顶层顶格的反引号围栏代码块的内容（与我们自己改写出来的图片），输出里没有
  ASCII 的 `![`，平台不会替模型去抓其中的地址；把每张图换成任意不超过 `IMAGE_DISPLAY_BYTES` 字节的文字之后，
  字节偏移 `VERBATIM_LIMIT_BYTES` 之后的部分连那里也没有。代价是：别处的代码（行内代码、波浪线围栏、缩进的或
  列表里的代码块）里举例的图片语法会显示成全角的感叹号。
- 只扣住还没确定的部分：结尾的 `!`、`\\`、`\\!`、`![` 或写了一半的语法，以及可能是链接地址或标题的部分（标题
  可以跨行，所以这一种会跨过换行；代码与原样放行的内容里只扣结尾的 `!`）。图片语法里不能有换行，说明、地址、标题与
  整段语法各有长度上限（`MAX_ALT_CHARS` 等），超过就不再当作图片；每个候选最多读 `MAX_SYNTAX_CHARS + 1` 个字符，所以很多个
  候选连在一起时整体仍是线性的。
- 一段（正文、代码或原样放行的内容）以 `!` 结尾、下一段换了分类时，这个 `!` 也先扣住，看了下一个字符才输出（是
  `[` 或不可见字符就换成全角；只有两个都是原样放行的内容时才留着）。所以「输出里没有 ASCII 的 `![`」不依赖块跟踪器
  对前后两个字符的分类一致；行首前缀里万一出现 `!`、`[`，也照样参与扫描。
- 无论怎样切分输入，输出拼起来都与一次性处理全文相同（「！」的替换与正文、代码的认定也一样）。

所以输出与单用 `MarkdownGuard` 的关系是：把每张图换回 guard 对同一段原文的输出之后，只有被换成「！」的 `!`
不同，其余每个字符都相同。

范围：这里管的是 **Markdown 的图片语法**，不处理原始 HTML（如正文里的 `<img src=…>`）。pyqbot 的实测
（errata「Markdown：实际支持的语法比文档多」）是 QQ 的 Markdown 不渲染 HTML 标签；模型本来就可以在正文里写
`<img>`，不是这里引入的；把正文与行内代码里的 `<img` 改掉会改坏讲解 HTML 的回答，而它会不会真的被加载并没有
证据。只有我们自己拼进输出的说明文字里的尖括号换成全角（`clean_alt()`），不让标签经由兜底文字或改写的图片语法
带进去。

已知的限制：

- 回答里出现过 `_Gate` 拿不准的写法之后，代码块里的 `![` 都显示成全角感叹号。常见的有：列表项里缩进的代码块、
  用 `\\(…\\)`、`\\[…\\]`、`$$` 写的公式、四个以上反引号的围栏（如里面套三个反引号的例子）、波浪线围栏、以 `<`
  开头的行、紧跟在一行正文后面（中间没有空行）的代码块。公式的标记在代码块里也算：代码里出现过 `$$`（如 shell 的
  进程号）、`\\(`、`\\[`（如正则表达式）之后，后面的 `![` 都显示成全角感叹号。
- 原样放行假定平台把还没有结束行、或结束行不规范的围栏仍当作代码（流式消息的每一个中间状态都是这样）：结束行
  不规范时，在它之前已经放行的内容收不回来。这一条靠 QQ 的实测确认（见 `_Gate`）。
- 原样放行的字节上限从一条回答的开头算起：机器人自己分条之后不重新计数（扫描器不知道分条），所以一条回答大约前
  `VERBATIM_LIMIT_BYTES` 字节之外的代码块里，`![` 都显示成全角感叹号。
- 说明里有行内代码的图片（`` ![`x`](地址) ``）不认：块跟踪器把其中的反引号算作代码，语法在那里被截断。
- 尖括号自动链接里的图片写法（`<https://example.org/![图](https://example.org/a.png)>`）照常认成图片，而
  CommonMark 认为那只是一个自动链接。
- 图片语法里不能有换行，这是有意的（只扣住有限的内容）。CommonMark 允许的这几种写法不认：说明里换行、地址前后
  换行（如 `![图](地址\\n (说明))`）、标题跨行。
- 被行内代码截断的链接地址或标题之后，到下一个空行为止的图片不认（如 ``[文档](地址 "使用 `x` 参数")。![图](地址)``，
  按 CommonMark 这张图会渲染）：为这种少见的写法在 `_tail` 里接着解析链接的后半段不值得。
- 引用式的图片（`![说明][标签]` 与别处的 `[标签]: 地址`）不认，感叹号换成全角。
- 图片语法里（说明、地址或标题）有没转义的 `|` 时不认：扫描器不知道自己在不在表格里，而表格会在那里把它拆进两个
  单元格。所以表格之外说明里带竖线的图片（`![A|B](地址)`）也不认；写成 `\\|` 的照常认。
- 表格是 CommonMark 之外的扩展，扫描器不跟踪表格的结构（哪几行是表格、有几列、表格在哪里结束），行内代码按
  CommonMark 配对。表格会在竖线处拆开单元格、每行重新开始，所以某个单元格里有一个单独的反引号时，它在这里可能与
  别的单元格或下一行的反引号配成一对，其中的图片写法可能被认出；表格特有的边界上（表格之后缩进的行、比表头多出来
  的单元格等）也可能把不会显示的图片写法认成图片。认出的图仍然要过白名单。
- 块引用与列表项嵌套超过 `MAX_CONTAINERS` 层之后，这条回答余下的部分都按代码处理（不认图片，也不再原样放行），
  只会少认；正常的文档远远到不了这么深。
- 图片或链接的标题里有不成对的反引号时，到这一段结束（下一个空行）为止的图片不认：块跟踪器把那个反引号当成了行内
  代码的开头，之后的分类与 CommonMark 正好相反，只好都不认（只会少认）。
- 配 `[` 时没有跟踪「链接里不能再有链接」：`[x [a](b)](![图](地址))` 里外层的 `[` 按 CommonMark 已经失效，`](`
  之后的图会渲染，这里仍当作链接的地址，不认（只会少认）。
- 下面几种少见的写法会多认（认出的图仍然要过白名单）：
  - 链接的地址或标题超过长度上限（`MAX_SYNTAX_CHARS`）时，超出的部分重新当作正文扫描，其中的图片写法可能被认成
    图片（按 CommonMark 它只是标题里的文字）。
  - 图片语法里面（说明、地址或标题里）再嵌着图片语法、链接或自动链接时（如 `![[x](地址a](地址b))](地址c)`、
    说明里有含 `]` 的 `<https://…]>`），认出的可能是嵌在里面的那一个，也可能都不认。
  - 引用定义的两种跨行写法：标签跨行（`[some` 换行 `label]: 地址 "标题"`）的不认作定义；标题跨行、其中有一行
    只有 `=`（或只有 `-`）的，在那一行提前结束了定义。它们的标题里的图片写法会被认出。
  - 尖括号自动链接的地址里有反引号时（如 ``<https://example.org/a`>``），块跟踪器把它当成行内代码的开头，同一段里
    之后的行内代码可能被认反，其中的图片写法可能被认出。
"""

import bisect
import itertools
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from html.entities import html5
from typing import NamedTuple
from urllib.parse import urlsplit

from .markdown_guard import MarkdownGuard, Segment, SegmentKind
from .media import known_host

# ---- 白名单 ----

DEFAULT_HOSTS: tuple[str, ...] = (
    # 入选的标准：这个后缀下只有站点自己（或按注册规则只有某一类机构）能提供内容、看到请求。
    # 政府与国际组织：政府后缀只收联合国安理会五个常任理事国的。
    "gov.cn",
    "gov",  # 美国
    "gov.uk",
    "gouv.fr",
    "gov.ru",
    "int",  # 条约设立的国际组织（世卫组织、气象组织、国际电联等）
    "un.org",  # 联合国
    "europa.eu",  # 欧盟机构
    # 百科、开放教材与数据
    "wikimedia.org",
    "openstax.org",
    "ourworldindata.org",
    # 科研与教育机构：edu 与 edu.cn 比政府后缀弱一档（大学会把子域名与个人主页分给院系和个人）。
    "edu",
    "edu.cn",
    "cas.cn",  # 中科院
    "cern.ch",
    "esawebb.org",
    "eso.org",
    "nobelprize.org",
    # 博物馆
    "dpm.org.cn",  # 故宫
    "chnmuseum.cn",  # 国博
    "louvre.fr",
    "metmuseum.org",
    "britishmuseum.org",
    # 国内官方媒体：图可能带水印，版权归各家所有。
    "news.cn",
    "xinhuanet.com",
    "people.com.cn",
    "chinadaily.com.cn",
)
"""内置的图站白名单（域名后缀，小写）。`gov`、`int`、`edu` 是单段的后缀，只在这里用；配置里追加的项要求至少两段。"""


def image_hosts(extra: Sequence[str] = ()) -> tuple[str, ...]:
    """图片下载的白名单：内置的 `DEFAULT_HOSTS`，后面接上配置里追加的（`[web_images] extra_hosts`，已校验），
    转成小写、去掉重复的。结果可以直接交给下载器（同附件下载的 `hosts`）。"""
    return tuple(dict.fromkeys((*DEFAULT_HOSTS, *(host.lower() for host in extra))))


def url_allowed(url: str, hosts: Sequence[str]) -> bool:
    """这个地址能不能下载：http(s)、不带用户信息、主机名是合法的域名并属于白名单 `hosts`（见 `image_hosts()`）。

    只是下载之前的预先检查，省得为名单外的地址发请求；下载器对每一跳（含重定向）另有完整的检查。
    """
    if "\\" in url or any(c.isspace() or not c.isprintable() for c in url):
        return False
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018  端口不是数字时这里抛 ValueError
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or "@" in parts.netloc or host is None:
        return False
    if "[" in parts.netloc or "]" in parts.netloc:  # 方括号里的是 IP 字面量（或别的），不是域名
        return False
    return known_host(host, hosts)


USER_AGENT = "ChatImageFetcher/1.0 (chat assistant that downloads images cited in its answers)"
"""下载图片时的 User-Agent：说明身份，不含个人或部署的信息。"""


# ---- 维基原图改写成缩略图 ----

WIKIMEDIA_THUMB_WIDTHS = (960, 500, 330, 250, 120)
"""维基原图改写成缩略图时依次尝试的宽度（维基的标准缩略图宽度）。缩略图不能比原图宽，事先又不知道原图多宽，所以
从大到小试；都不行时最后试原地址。"""

_WIKIMEDIA_ORIGINAL = re.compile(
    r"/wikipedia/(?P<project>[a-z0-9-]+)/(?P<a>[0-9a-f])/(?P<ab>(?P=a)[0-9a-f])/(?P<name>[^/]+)"
)
# 能出缩略图、改写后格式不变的扩展名；SVG 的缩略图是 PNG。其余（TIFF、PDF、视频等）的缩略图另有命名规则，不改写。
_THUMBNAIL_EXTENSIONS = frozenset(("jpg", "jpeg", "png", "gif", "webp", "svg"))


def download_plan(url: str, hosts: Sequence[str]) -> list[str]:
    """模型写的图片地址 `url` 依次尝试下载的地址；不能下载时是空列表。

    先用 `url_allowed()` 检查原地址，不通过就不改写、不下载。通过了再改写：
    `upload.wikimedia.org/wikipedia/<项目>/<a>/<ab>/<文件名>` 这样的维基原图改写成
    `https://thumb.wikimedia.org/wikipedia/<项目>/thumb/<a>/<ab>/<文件名>/<宽>px-<文件名>`，宽度按
    `WIKIMEDIA_THUMB_WIDTHS` 从大到小各一个（SVG 的再加 `.png`；去掉查询串与片段，文件名里的百分号编码原样保留），
    最后是原地址。其余地址（包括已经是缩略图的）原样作为唯一的一个。返回的每个地址也都通过 `url_allowed()`。
    """
    if not url_allowed(url, hosts):
        return []
    return [candidate for candidate in _download_candidates(url) if url_allowed(candidate, hosts)]


def _download_candidates(url: str) -> list[str]:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return [url]
    if (
        parts.scheme.lower() not in ("http", "https")
        or parts.hostname != "upload.wikimedia.org"
        or port is not None
        or "@" in parts.netloc
    ):
        return [url]
    match = _WIKIMEDIA_ORIGINAL.fullmatch(parts.path)
    if match is None:
        return [url]
    name = match["name"]
    stem, dot, extension = name.rpartition(".")
    extension = extension.lower()
    if not (stem and dot) or extension not in _THUMBNAIL_EXTENSIONS:
        return [url]
    suffix = ".png" if extension == "svg" else ""
    base = (
        f"https://thumb.wikimedia.org/wikipedia/{match['project']}/thumb/"
        f"{match['a']}/{match['ab']}/{name}"
    )
    return [*(f"{base}/{width}px-{name}{suffix}" for width in WIKIMEDIA_THUMB_WIDTHS), url]


# ---- 尺寸 ----

# JPEG 里带图像尺寸的帧头（SOF）标记：C0–CF，去掉 C4（DHT）、C8（保留）、CC（DAC）。
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# JPEG 里没有长度字段的标记：TEM、RST0–7、SOI。
_JPEG_STANDALONE = frozenset((0x01, *range(0xD0, 0xD8), 0xD8))


def image_size(data: bytes) -> tuple[int, int] | None:
    """按文件头读出图片的（宽，高）像素，支持 PNG、JPEG、GIF、WebP（VP8、VP8L、VP8X）。认不出、数据不完整或
    宽高为 0 时返回 `None`，不抛异常。JPEG 的帧头可能在 EXIF 之后，要给到它为止的数据。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        size = _png_size(data)
    elif data.startswith(b"\xff\xd8"):
        size = _jpeg_size(data)
    elif data.startswith((b"GIF87a", b"GIF89a")):
        size = _le(data, 6, 2), _le(data, 8, 2)
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        size = _webp_size(data)
    else:
        return None
    if size is None:
        return None
    width, height = size
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return width, height


type _Size = tuple[int | None, int | None]


def _le(data: bytes, start: int, length: int) -> int | None:
    """小端无符号整数；数据不够时为 `None`。"""
    if start + length > len(data):
        return None
    return int.from_bytes(data[start : start + length], "little")


def _be(data: bytes, start: int, length: int) -> int | None:
    """大端无符号整数；数据不够时为 `None`。"""
    if start + length > len(data):
        return None
    return int.from_bytes(data[start : start + length], "big")


def _png_size(data: bytes) -> _Size | None:
    # 第一个块必须是 IHDR，长度固定是 13。
    if data[12:16] != b"IHDR" or _be(data, 8, 4) != 13:
        return None
    return _be(data, 16, 4), _be(data, 20, 4)


def _jpeg_size(data: bytes) -> _Size | None:
    i = 2
    while True:
        if i >= len(data) or data[i] != 0xFF:
            return None
        while i < len(data) and data[i] == 0xFF:  # 标记前可以有多个填充的 0xFF
            i += 1
        if i >= len(data):
            return None
        marker = data[i]
        i += 1
        if marker in _JPEG_STANDALONE:
            continue
        if marker in (0x00, 0xD9, 0xDA):  # 不是标记，或在帧头之前就到了结尾、图像数据
            return None
        length = _be(data, i, 2)
        if length is None or length < 2:
            return None
        if marker in _JPEG_SOF:
            # 段的内容：长度（2）、精度（1）、高（2）、宽（2）、分量数（1）……；声明的长度要够得着这些。
            # 每个分量的记录 3 字节，要放得进声明的段长。
            if length < 8 or i + 8 > len(data) or data[i + 7] < 1 or length < 8 + 3 * data[i + 7]:
                return None
            return _be(data, i + 5, 2), _be(data, i + 3, 2)
        i += length


def _webp_size(data: bytes) -> _Size | None:
    riff_size, chunk_size = _le(data, 4, 4), _le(data, 16, 4)
    if riff_size is None or chunk_size is None:
        return None
    # 读到的字段要落在 RIFF 声明的文件长度（从第 8 字节起算）与第一个块声明的长度（从第 20 字节起算）之内。
    limit = min(8 + riff_size, 20 + chunk_size)
    match data[12:16]:
        case b"VP8 ":
            # 帧标签（3 字节）之后是起始码 9D 01 2A，再是各 14 位的宽、高（高 2 位是缩放）。
            if limit < 30 or data[23:26] != b"\x9d\x01\x2a":
                return None
            width, height = _le(data, 26, 2), _le(data, 28, 2)
            if width is None or height is None:
                return None
            return width & 0x3FFF, height & 0x3FFF
        case b"VP8L":
            # 签名 0x2F 之后：宽减 1、高减 1 各 14 位。
            if limit < 25 or data[20:21] != b"\x2f":
                return None
            bits = _le(data, 21, 4)
            if bits is None:
                return None
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        case b"VP8X":
            # 标志（1）与保留（3）之后：画布的宽减 1、高减 1 各 24 位。
            if limit < 30:
                return None
            width, height = _le(data, 24, 3), _le(data, 27, 3)
            if width is None or height is None:
                return None
            return width + 1, height + 1
        case _:
            return None


def display_size(width: int, height: int, limit: int) -> tuple[int, int]:
    """图片语法里写的（宽，高）：按原图的比例，长边不超过 `limit`，比它小的不放大，短边至少 1。

    Raises:
        ValueError: 宽、高或上限不是正整数。
    """
    if width < 1 or height < 1 or limit < 1:
        raise ValueError("宽、高与上限都要是正整数")
    if max(width, height) <= limit:
        return width, height
    if width >= height:
        return limit, max(1, (height * limit + width // 2) // width)
    return max(1, (width * limit + height // 2) // height), limit


# ---- 流式扫描 ----

MAX_ALT_CHARS = 200
"""说明文字（方括号之间）最多这么多个字符；没写完就超过的，当作文字放行。"""
MAX_URL_CHARS = 2048
"""地址最多这么多个字符。"""
MAX_TITLE_CHARS = 200
"""标题（引号之间）最多这么多个字符。"""
MAX_SYNTAX_CHARS = MAX_ALT_CHARS + MAX_URL_CHARS + MAX_TITLE_CHARS + 32
"""整段图片语法最多这么多个字符，主要限住各部分之间的空白。"""

DEFUSED_BANG = "！"
"""正文里紧跟着 `[`、又没有被认成图片的 `!` 换成它，平台就不会把那段文字当成图片去抓。"""

MAX_URL_PARENS = 32
"""裸地址里圆括号最多嵌套这么多层（同 cmark），更深的不算图片。"""

VERBATIM_LIMIT_BYTES = 12000
"""一条流式消息的前这么多字节（UTF-8）里，顶格反引号围栏里的 `![` 才可以原样放行。平台把超过约 15000 字节的流式消息
不看结构地硬切成几条气泡（实测最早在 14387 字节处，见 `splitting.py`），切口落在围栏里时，后一条从围栏中间开始，
里面的 `![` 就成了正文；这么多字节之前平台不会切。机器人自己分条也只在 `splitting.SPLIT_SOFT_BYTES` 之后，所以它不大于
那个数（扫描器不知道分条，分条之后不重新计数）。"""
IMAGE_DISPLAY_BYTES = 1024
"""一张图显示时最多占这么多字节（UTF-8）：`render_image()` 的结果超过它就不用（抛 `ValueError`），`fallback_text()`
的结果按构造不会超过。扫描器按这个数计算每张图之后的字节偏移。"""

_URL_SCHEMES = ("http://", "https://")
_SPECIAL = re.compile(r"[!\\\[\]]")
_LINE_END = re.compile(r"[\r\n]")
_URL_STOP = re.compile(r"[()\\|\x00-\x20\x7f]")  # 裸地址里要停下来看的字符
_ANGLE_STOP = re.compile(r"[<>\r\n\\|]")  # 尖括号地址里要停下来看的字符
_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")  # 能被反斜杠转义的 ASCII 标点
# 字符引用解码出这些（空白与控制字符）时地址不算图片。
_DECODED_BAD = re.compile(r"[\x00-\x20\x7f-\x9f\s]")
# CommonMark 的反斜杠转义与字符引用，一遍处理（被转义的 `&` 不再当作字符引用的开头）：字符引用只认以分号结尾的，
# 十进制最多 7 位、十六进制最多 6 位，名字查 HTML5 的实体表。
_ESCAPE_OR_REFERENCE = re.compile(
    r"\\([!-/:-@\[-`{-~])|&(?:#([0-9]{1,7})|#[xX]([0-9a-fA-F]{1,6})|([A-Za-z][A-Za-z0-9]{0,31}));"
)


@dataclass(frozen=True, slots=True)
class ImageRef:
    """模型写的一张图。

    Attributes:
        alt: 说明文字，已去掉反斜杠转义，没有清理过（显示前用 `render_image()` / `fallback_text()`）。
        url: 地址（不含尖括号），字符引用已按 CommonMark 解码（如 `&amp;` 成了 `&`），以 `http://` 或 `https://`
            开头（不分大小写）；字符引用解码出空白或控制字符的不算图片。尖括号地址里字面的空格写成 `%20`（同参照
            解析器），别的空白与控制字符不算图片。
        raw: 这段图片语法的原文，即模型写的样子（不经 guard 改写）。
    """

    alt: str
    url: str
    raw: str


type ScanItem = str | ImageRef


class _Verdict(Enum):
    NO = "no"  # 不是图片
    MORE = "more"  # 要更多输入才能确定


# `\n`、`\r` 之外，有的解析方式也当作行尾的字符（Python 的 `str.splitlines()` 认的那些）。
_LINE_BREAKS = frozenset("\v\f\x1c\x1d\x1e\x85" + chr(0x2028) + chr(0x2029))
# 格式字符（Cf）之外，显示时看不见、有的解析方式会忽略的字符：组合用字形连接符、韩文填充符。
_FILLERS = frozenset(map(chr, (0x034F, 0x115F, 0x1160, 0x3164, 0xFFA0)))


def _invisible(c: str) -> bool:
    """有的解析方式会忽略的不可见字符：格式字符（Unicode 类别 Cf，如 BOM、零宽空格、零宽连接符、方向控制符、软连字符），
    变体选择符，蒙古文自由变体选择符，以及 `_FILLERS`。"""
    if c.isascii():
        return False
    code = ord(c)
    return (
        unicodedata.category(c) == "Cf"
        or 0xFE00 <= code <= 0xFE0F
        or 0xE0100 <= code <= 0xE01EF
        or 0x180B <= code <= 0x180F
        or c in _FILLERS
    )


def _defuses(following: str) -> bool:
    """不认作图片开头的 `!` 后面接的是 `following` 时，要不要换成全角：后面是 `[`，或是不可见字符（`_invisible()`；
    有的解析方式忽略它，`!` + 零宽空格 + `[` 按那种方式仍是 `![`）。只看后面一个字符。"""
    return following[:1] == "[" or (following != "" and _invisible(following[0]))


def _defuse_bangs(text: str) -> str:
    """`text` 里后面跟着 `[` 或不可见字符的 `!` 都换成全角（结尾的 `!` 不动，由调用方看下一个字符）。"""
    if "!" not in text:
        return text
    parts: list[str] = []
    start = 0
    position = text.find("!")
    while 0 <= position < len(text) - 1:
        if _defuses(text[position + 1]):
            parts.append(text[start:position])
            parts.append(DEFUSED_BANG)
            start = position + 1
        position = text.find("!", position + 1)
    parts.append(text[start:])
    return "".join(parts)


def _blank(c: str) -> bool:
    """宽松的解析方式可能当作空白的字符：Unicode 的空白，以及不可见字符（`_invisible()`）。"""
    return c.isspace() or _invisible(c)


class _Gate:
    """原样放行的闸门（只回答一件事：一个字符能不能原样输出，见模块 docstring 的分工）。与 `_BlockTracker` 各自独立地
    看原文、互不调用。它不模拟哪一家解析器，只认一种「规范形状」：到目前为止的全文都是规范形状时，才放行顶格反引号
    围栏的内容；一旦出现任何一种拿不准的写法（各家解析器可能看法不一、或者 QQ 另有语法的），这条回答从此不再放行
    （`_off`，不可恢复）。宁可不放行。

    按 `\\n` 分行（行尾的 `\\r\\n` 算一个行尾）。围栏之外逐行看：

    - **规范的开头行**：从第 0 列开始（前面没有任何字符，包括空格与 BOM），恰好 3 个反引号，之后到行尾没有反引号；
      上一行是空行（只有空格或制表符也算），或它就是全文的第一行。进入围栏。开头行本身不放行。上一行有内容时不算：
      有的实现不让围栏打断前一段（实测 Ruby 的 RDoc::Markdown），那里这对反引号不是围栏，之后的围栏也会错位，所以
      按下一条从此不再放行。4 个以上的不算：有的实现不认（实测 Ruby 的 RDoc::Markdown 不认四个反引号的围栏），
      用正则成对匹配三个反引号的解析器又会把 6 个以上的配成一对空代码块；而四、五个反引号的围栏几乎只在里面要套三个
      反引号的行时才用，那一行本来就让闸门关闭，放行它们换来的很少。
    - 不是规范的开头行，而这一行里任何位置有连续 3 个以上的反引号：`_off`。缩进的围栏、列表与引用里的围栏、信息串
      带反引号的行、行中间的三个反引号都归到这里。
    - 这一行里任何位置有 `~~~`：`_off`，规范的开头行（信息串里）也一样。波浪线围栏各家支不支持不一，有的还拿它
      结束反引号围栏。
    - 这一行以 `<` 开头（前面可以有容器标记与空白，空白按 `_blank()` 算，含 Unicode 空白与不可见字符）：`_off`。
      CommonMark 的 HTML 块里的反引号行不是围栏。
    - 找 `$$`、`\\[`、`~~~`、连续的反引号时跳过不可见字符（`_invisible()`）：有的解析方式忽略它们，`$` 与 `$` 之间
      隔着零宽空格也是 `$$`。

    围栏之内逐行看：

    - **规范的结束行**：从第 0 列开始，恰好 3 个反引号，之后到行尾只有空格或制表符。离开围栏。
    - 不是规范的结束行，而这一行里任何位置有连续 3 个以上的反引号，或有 `~~~`：`_off`。缩进的结束行、4 个以上反引号
      的行、反引号后面跟着别的字符的行、行中间的反引号串都归到这里（有的解析器把它们当作结束行或配对的标记）。
    - 其余是内容，可以放行。

    全文范围：

    - 公式的标记（`$$`、`\\[`、`\\]`、`\\(`、`\\)`）不论在围栏里还是围栏外，一出现就从这个字符起 `_off`（同一行里
      它之前已经放行的内容不受影响）。QQ 支持 `$$…$$`、`\\[…\\]` 块公式，按「公式先取走、可以跨行配对」的方式，
      两个标记之间的围栏行会被吞掉，吞掉的开头行与结束行不一定成对，之后的围栏就错位了。所以放行的内容都在全文第一个
      公式标记之前。
    - 出现单独的 `\\r`（后面不是 `\\n`），或 `\\v`、`\\f`、`\\x1c`–`\\x1e`、`\\x85`、U+2028、U+2029（`_LINE_BREAKS`，
      有的解析方式把它们当作行尾）：`_off`。各家对它们算不算行尾看法不一，围栏里外都一样处理。
    - 全文第一个非空行（空行：只有 `_blank()` 的字符），去掉开头的空白与不可见字符之后以 `---` 或 `+++` 开头：
      `_off`。有的解析器把它当作 front matter 的开头，有的在认之前先去掉行两侧的 Unicode 空白，所以按最宽的算
      （`--- yaml`、`----`、前后有不换行空格的都算）。

    为什么放行的内容按各种别的解析方式看也在代码块里。先看规范形状的一个事实：放行的字符之前，连续 3 个以上的反引号
    只出现在规范的开头行与结束行里，每处恰好 3 个（别处的、更长的一出现就 `_off`），开头行与结束行从第一个开头行起
    严格交替；之前也没有 `~~~`、公式标记、单独的 `\\r` 与别的行尾字符、行首的 `<`、首行的 `---`/`+++`。

    - CommonMark（参照的 markdown-it）：只有规范的开头行能开始围栏，围栏里只有规范的结束行能结束它，与闸门的跟踪
      一致。
    - 用正则成对匹配三个反引号（行中间也配对）：全文里的「三个反引号」恰好是这些开头行与结束行，按出现的顺序第 1、
      2 处配成一对、第 3、4 处配成一对……正好是每个开头行配它的结束行，内容都在对里。
    - 结束行可以带别的字、或不少于开头的个数就算结束：围栏里除了它的结束行没有别的 3 个以上反引号的行，结束的仍是
      同一行。
    - 只认三个反引号的围栏、不认更长的：放行的围栏都是三个反引号的。
    - 不让围栏打断前一段（开头行前面要有空行）：放行的围栏的开头行前面都是空行或全文开头。
    - 不认波浪线围栏，或拿 `~~~` 行结束反引号围栏：放行之前没有 `~~~`。
    - 公式先取走（可以跨行配对）：放行的内容都在第一个公式标记之前，被取走的部分从那个标记起，碰不到它们与它们
      前面的围栏行。
    - 单独的 `\\r` 或别的字符也算行尾：放行之前没有这些字符，分出来的行与按 `\\n` 分的相同。
    - 先剥掉 front matter：第一个非空行不以 `---`、`+++` 开头（去掉两侧的空白之后也不），不会剥掉什么。
    - HTML 块：放行之前没有以 `<` 开头的行（前面的空白按宽松的算）。
    - 对空白更宽松（把 Unicode 空白、BOM、零宽字符当作空白或直接忽略）：开头行与结束行要求第 0 列、之后只有空格或
      制表符，从严的方向不会反过来（结束行后面跟着 Unicode 空白时不是规范的结束行，行里有 3 个以上的反引号，`_off`；
      前面有 Unicode 空白或 BOM 的开头行同理）；`<`、首行、`~~~`、公式标记与连续的反引号按宽松的算法判断（跳过
      不可见字符）。
    - 平台把长消息硬切成几条：见 `VERBATIM_LIMIT_BYTES`（扫描器另外管）。

    放行依赖一个去不掉的前提：平台把还没有结束行（或结束行不规范）的围栏仍当作代码。流式消息的每一个中间状态都是
    没闭合的围栏；结束行到来之前已经放行的内容收不回来（结束行不规范时闸门之后才关闭）。把内容扣到确认结束行再输出
    会让带 `![` 的代码块要等写完才显示，所以不这样做。这一条靠 QQ 的实测确认。这里列出的解析方式之外，别的实现在
    围栏上的其它差异不再追。

    逐个字符给出结论，不暂存整行，状态只有几个计数，所以没有长度上限，结论只取决于这个字符之前（含）的原文，不随
    切分变化，也不需要回头改已经给出的结论：

    - 围栏外的字符都不放行，所以围栏外的规则到行尾才用上也不晚（它们只影响之后的行）；开头行在围栏外，同样不放行。
    - 围栏里的一行：从第 0 列起的反引号还没结束时不放行（可能是结束行）。这串有 3 个以上时，这一行只可能是结束行或
      拿不准的写法，整行都不放行，到行尾再看：不是规范的结束行就 `_off`。这串不到 3 个时这一行是内容，之后的字符逐个
      放行，直到连续的第三个反引号或波浪线：从它起 `_off`。在它之前的字符（含这串的前两个）对任何解析器都还在代码块
      里；这串的前两个字符是反引号或波浪线，原样输出与按代码输出本来就没有区别。
    - 行尾的 `\\r`、`\\n` 不放行（同样没有区别）；`\\r` 后面不是 `\\n` 时，从后面那个字符起 `_off`。
    """

    _PREFIX = frozenset(">*+-0123456789.)")  # 行首可能是容器标记的字符（空白另由 `_blank()` 判断）
    _MATH = frozenset(("$$", "\\[", "\\]", "\\(", "\\)"))  # 公式的标记

    def __init__(self, *, off: bool = False) -> None:
        self._off = off  # 拿不准了：不再放行
        self._fence = 0  # 所在围栏开头行的反引号个数；0 表示不在围栏里
        self._first = True  # 还没有过非空行（空行：只有 `_blank()` 的字符）
        self._after_blank = True  # 上一行只有空格与制表符，或这是全文的第一行：规范的开头行要的
        self._cr = False  # 上一个字符是 `\r`：紧接着的 `\n` 与它同属一个行尾
        self._new_line()

    def _new_line(self) -> None:
        self._lead = 0  # 从第 0 列起的一串反引号有几个
        self._leading = True  # 到目前为止这一行只有从第 0 列起的那串反引号
        self._blank_after = True  # 那串反引号之后只有空格与制表符
        self._tick_after = False  # 那串反引号之后有反引号
        self._run_char = ""  # 正在读的一串反引号或波浪线，与它的长度
        self._run = 0
        self._triple = False  # 这一行有连续 3 个以上的反引号或波浪线
        self._previous = ""  # 这一行的上一个字符（不可见字符不算），找 `$$`、`\[` 这类公式的标记
        # 围栏外：是否还在行首的缩进与容器标记里，有没有拿不准的写法。
        self._prefix = True
        self._risky = False
        # 还没有过非空行时：这一行去掉开头的空白与不可见字符之后的前 3 个字符（不可见字符不算）。
        self._front = ""
        self._content = False  # 这一行有空白（`_blank()`）以外的字符
        self._spaces_only = True  # 这一行只有空格与制表符

    def mark(self, text: str) -> list[bool]:
        """依次处理 `text` 的每个字符，返回每个字符能不能原样输出。"""
        return [self._char(c) for c in text]

    def _char(self, c: str) -> bool:
        if self._cr:
            self._cr = False
            if c == "\n":
                return False
            self._off = True  # 单独的 `\r`
        if c in "\r\n":
            self._end_line()
            self._cr = c == "\r"
            return False
        if c in _LINE_BREAKS:
            self._off = True  # 有的解析方式把它当作行尾
        invisible = _invisible(c)
        if self._leading and c == "`":
            self._lead += 1
        else:
            self._leading = False
            self._tick_after = self._tick_after or c == "`"
            self._blank_after = self._blank_after and c in " \t"
        if invisible:
            pass  # 有的解析方式忽略它：不打断一串反引号或波浪线
        elif c in "`~":
            self._run = self._run + 1 if c == self._run_char else 1
            self._run_char = c
        else:
            self._run_char, self._run = "", 0
        self._triple = self._triple or self._run >= 3
        if not _blank(c):
            self._content = True
        if c not in " \t":
            self._spaces_only = False
        if not invisible:  # 有的解析方式忽略不可见字符：`$` 与 `$` 之间隔着它也算 `$$`
            if self._previous + c in self._MATH:
                self._off = True  # 公式的标记：围栏里外都从这个字符起不再放行
            self._previous = c
        if self._fence:
            self._fence_char()
            return not (self._off or self._leading or self._lead >= 3)
        self._outside_char(c, invisible=invisible)
        return False

    def _fence_char(self) -> None:
        # 从第 0 列起有 3 个以上反引号的行到行尾再看（`_end_line()`），在那之前它的字符都不放行。
        if not self._leading and self._lead < 3 and self._triple:
            self._off = True

    def _outside_char(self, c: str, *, invisible: bool) -> None:
        if not invisible and self._run_char == "~" and self._run >= 3:
            self._risky = True  # 规范的开头行里（信息串）也算
        if self._prefix and not (c in self._PREFIX or _blank(c)):
            self._prefix = False
            self._risky = self._risky or c == "<"
        if (
            self._first
            and len(self._front) < 3
            and not invisible
            and (self._front or not _blank(c))
        ):
            self._front += c

    def _end_line(self) -> None:
        if self._fence:
            if self._lead >= 3:
                if self._lead == self._fence and self._blank_after:
                    self._fence = 0
                else:
                    self._off = True
        else:
            if self._lead == 3 and not self._tick_after and self._after_blank:
                self._fence = self._lead
            elif self._triple:
                self._off = True
            front = self._first and self._front in ("---", "+++")
            if self._risky or front:
                self._off = True
        if self._content:
            self._first = False
        self._after_blank = self._spaces_only
        self._new_line()


class _Fence(Enum):
    """跟踪器眼里的一个字符在哪里。"""

    NONE = "none"  # 正文
    PREFIX = (
        "prefix"  # 行首的容器前缀（缩进与块引用的 `>`）：正文的一部分，但跨行的链接语法里不算字符
    )
    TOP = "top"  # 开头行顶格的顶层反引号围栏的内容：可以原样放行
    CODE = "code"  # 别的代码：别的围栏（含开头行）、缩进代码块、行内代码、链接引用定义冒号之后：不认图片


class _Container(NamedTuple):
    kind: str  # "quote" 或 "item"
    # 列表项的内容要缩进多少列，相对于外一层容器的内容开始处（块引用的前缀在各行可以不同，所以不能记绝对的列）；
    # 块引用不用。
    width: int


_BIG = 1 << 30

MAX_CONTAINERS = 32
"""块跟踪器最多跟踪这么多层容器（块引用与列表项）。再深就进入保守的状态，这条回答余下的部分都按代码处理（见
`_BlockTracker.mark()`）：不然一长串 `>` 会让容器的栈一直长下去。正常的文档远远到不了这么深。"""


class _BlockTracker:
    """按 CommonMark（以参照的 markdown-it 为准）的块结构逐个字符地判断每个字符在哪里，直接看原文，不靠 guard。跟踪
    块引用与列表项（容器）、任何一层容器里的围栏代码块与缩进代码块、段落与标题里的行内代码、链接引用定义。状态是容器的
    栈与几个计数，不暂存整行，所以没有长度上限，结论不随行长或切分变化。

    - 每行先按顺序匹配开着的容器：块引用要 `>`（续行的 `>` 前面缩进多少都认，同 markdown-it；其后的一个空格不算
      缩进），列表项要缩进到它的内容列（相对外一层容器的内容开始处；空行也算匹配上）。没匹配上的容器，这一行是段落的
      续行时留着（懒惰续行），否则关掉。判断列表项里的段落的续行是不是块的开头时，看相对列表项内容的缩进（可以是负数，
      同 markdown-it）；它在外层缩进了 4 列以上时，是块的开头就是外层的缩进代码。
    - 然后开新的容器：`>`（相对缩进 0–3）；列表标记（`-`、`+`、`*`，或 1–9 位数字加 `.`、`)`，之后是空白或行尾；
      内容列在标记之后 1–4 列空白处，空白有 5 列以上或标记之后就是行尾时在标记之后一列）。在同一层容器里打断段落的
      列表项不能是空的，有序的要从 1 开始。空的列表项紧接着空行就结束。分隔线先于列表项判断：从某个列表标记（或内容的
      开头，相对缩进 0–3）起到行尾只有 `-`、`*`、`_` 之一（至少三个）与空白时，从那里起是分隔线，撤回从那里起这一行
      开的列表项（`- - -` 整行是分隔线，`* - - -` 是列表项里的分隔线）。
    - 再看这一行是什么：围栏的开头（相对缩进 0–3，n ≥ 3 个反引号或波浪线，反引号围栏的信息串里没有反引号；到行尾才
      确定，这一行先按代码算）、缩进代码（相对缩进至少 4 列，不在段落里）、ATX 标题、setext 标题的下划线（相对缩进
      0–3）、链接引用定义、空行，其余是段落。
    - 围栏之后的各行，在它所在的各层容器都匹配上时是内容；结束行相对缩进 0–3、至少 n 个同一种字符、之后只有空格或
      制表符。容器没匹配上时围栏随之结束，围栏不能懒惰续行。
    - 链接引用定义：不在段落里（或紧跟在定义之后）的一行以 `[标签]:` 开头，之后按 CommonMark 检查地址与标题（地址可以
      在下一行，标题可以在下一行、可以跨行）。冒号之后按代码算；检查不通过时它其实是段落的文字，行内代码照常配对。
    - 行内代码：段落与标题里没有被反斜杠转义的一串反引号开始，到同样长的下一串结束；还没结束的部分都按代码算（没配上
      对的按 CommonMark 是普通文字，这样算只会少认）。新的段落或标题开始时清掉。拿不准是不是段落的续行时（链接引用
      定义、像块开头的续行），反引号照常配对、字符按代码算。
    - 行尾：`\\n`、`\\r\\n`、单独的 `\\r`。行尾的反斜杠不转义下一行的字符。
    - 只有顶层、开头行从第 0 列开始的反引号围栏的内容是 `TOP`。行首（容器标记之后）以 `<` 或 `$$` 开头的行出现之后，
      不再给出 `TOP`：CommonMark 的 HTML 块与 QQ 支持的 LaTeX 公式块里可以有顶格的反引号行，它们在那里不是围栏。
    - 原则：拿不准时，围栏、行内代码、定义这类「里面是代码或属性」的按代码算，不认图片；`TOP` 只在确定时给出。
    - `TOP` 只是原样放行要的三方同意之一（见模块 docstring 的分工）；这里判断错了，只影响哪里认图片，不会让 `![`
      原样漏出去。
    """

    def __init__(self) -> None:
        self._containers: list[_Container] = []
        self._overflow = False  # 容器嵌套超过了 `MAX_CONTAINERS` 层：之后都按代码算
        # 与容器栈同步：块引用在栈里的位置（从小到大），各层宽度的前缀和（`_widths[k]` 是前 k 层之和）。有了它们，
        # 每行要看「没匹配上的那几层」时不用逐层去数，开销不随嵌套的深度增长。
        self._quotes: list[int] = []
        self._widths: list[int] = [0]
        self._fence = ""  # 所在围栏的标记字符；空串表示不在围栏里
        self._fence_length = 0
        self._fence_top = False
        self._code = False  # 最里层的容器里开着缩进代码块
        self._para = False  # 最里层的容器里开着段落（或拿不准、按段落接着配对行内代码）
        self._settled = True  # 开着的段落是真的段落：可以懒惰续行
        self._off = False  # 见过 `<`、`$$` 开头的行：不再给出 `TOP`
        self._span = 0  # 所在行内代码开头那串反引号的长度；0 表示不在里面
        self._run = 0  # 正在读的一串反引号的长度
        self._escaped = False  # 上一个字符是正文里没有被转义的反斜杠
        self._cr = False  # 上一个字符是 `\\r`：紧接着的 `\\n` 与它同属一个行尾
        self._cr_where = _Fence.NONE
        self._empty_item = -1  # 刚开的空列表项让容器栈有多深；-1 表示没有
        # 链接引用定义的后续行：dest（下一行是地址）、title（下一行若以引号或括号开头是标题）、more（标题还没完）；
        # 空串表示没有。`_after_definition` 表示上一行是定义：下一行还可以是定义。
        self._definition_next = ""
        self._definition_closer = ""
        self._after_definition = False
        self._new_line()

    def _new_line(self) -> None:
        self._col = 0
        self._base = 0  # 当前这一层容器的内容列
        self._matched = 0  # 这一行匹配上的已有容器数
        self._gt = False  # 刚读过 `>`：之后的一个空格（或制表符的一列）不算缩进
        self._phase = "match"  # match（匹配已有容器）、open（开新容器）、leaf（这一行的内容）
        # 这一行是什么：fence、code、opener、atx、heading、definition、maybe（像块开头的续行）、text
        self._leaf = ""
        self._lazy = False  # 没匹配上的容器因为懒惰续行留着
        # 列表标记：读到哪里（bullet、digits、delim、spaces）、从哪一列开始、标记在哪一列结束、序号。
        self._marker = ""
        self._marker_col = 0
        self._marker_end = 0
        self._number = 0
        self._digits = 0
        # 围栏的开头或结束行。
        self._mark = ""
        self._marks = 0
        self._after = False
        self._tick_in_info = False
        self._mark_col = 0
        self._closing = False
        self._hashes = 0
        # 链接引用定义这一行读到哪里：label、colon、ws1、angle、bare、after、ws2、title、after_title、bad，以及
        # 标签里有没有字、地址里的括号层数、反斜杠转义、标题的结束字符。
        self._definition = ""
        self._label = 0
        self._label_text = False
        self._parens = 0
        self._def_escape = False
        self._closer = ""
        self._def_code = False  # 已过了冒号（或是后续行）：之后都按代码算
        self._def_more = False  # 这一行是定义的后续行
        self._def_opener = False  # 后续行同时可能是围栏的开头
        self._def_atx = False  # 后续行同时可能是 ATX 标题
        # 像块开头的续行：读到哪里（run、hash、rule）与计数；结论（None 表示还没定）。
        self._maybe = ""
        self._maybe_count = 0
        self._maybe_block: bool | None = None
        self._dollar = 0  # 1：这一行的内容以 `$` 开头，2：等第二个 `$`
        self._risky = False
        # 分隔线：从这一行的每个列表标记、以及内容的开头起，到行尾是不是只有 `-`、`*`、`_` 之一（至少三个）与空白。
        # 每项是（从这里起是分隔线时要撤回的容器从栈里的哪一层开始，字符，个数）；不再可能的项删掉。按 CommonMark，
        # 外层先判断分隔线、再开列表项，所以 `- - -` 整行是分隔线，`* - - -` 是列表项里的分隔线。
        # 每加一项之后紧接着读那个字符，`_shape()` 读完一个字符只留起支配作用的一项，所以最多两项，每个字符的开销不随
        # 列表标记的个数增长。
        self._rules: list[tuple[int, str, int]] = []
        # 一串连续的 `=` 或 `-`，之后只有空白（setext 标题的下划线：中间不能有空白，分隔线才可以）；这一行内容的相对
        # 缩进不到 4 列（分隔线与下划线要求的）。`_underline_gap` 表示那串之后已经有过空白。
        self._underline = ""
        self._underline_gap = False
        self._shape_ok = True

    # ---- 逐个字符 ----

    def mark(self, text: str) -> list[_Fence]:
        """依次处理 `text` 的每个字符，返回每个字符在哪里。容器嵌套超过 `MAX_CONTAINERS` 层之后，余下的都按代码算
        （不认图片，也不会原样放行），状态不再增长；只会少认。"""
        marks: list[_Fence] = []
        for c in text:
            where = _Fence.CODE if self._overflow else self._char(c)
            marks.append(_Fence.CODE if self._overflow else where)  # 触发上限的那个字符也按代码算
        return marks

    def _char(self, c: str) -> _Fence:
        if self._cr:
            self._cr = False
            if c == "\n":
                return self._cr_where
        if c in "\r\n":
            where = self._fence_where() if self._leaf == "fence" else _Fence.NONE
            self._end_line()
            if c == "\r":
                self._cr, self._cr_where = True, where
            return where
        if c in " \t" and self._underline in ("=", "-"):
            self._underline_gap = True  # 之后再有 `=`、`-` 就不是下划线了（见 `_shape()`）
        if self._phase == "match" and not self._match(c):
            return _Fence.PREFIX
        if self._phase == "open":
            if self._marker:
                if not self._marker_char(c):
                    return _Fence.PREFIX if c in " \t>" else _Fence.NONE
            elif not self._open(c):
                return _Fence.PREFIX
        return self._content(c)

    def _advance(self, c: str) -> None:
        """空格或制表符：列号前进（制表符对齐到 4 的倍数）；紧跟 `>` 的那一列不算缩进。"""
        self._col = self._col + 1 if c == " " else self._col + 4 - self._col % 4
        if self._gt:
            self._base += 1
            self._gt = False

    def _match(self, c: str) -> bool:
        """匹配开着的容器；返回 `True` 表示这个字符要接着处理（已过了这一阶段）。"""
        if c in " \t":
            self._advance(c)
            return False
        self._gt = False
        while self._matched < len(self._containers):
            kind, width = self._containers[self._matched]
            if kind == "item":
                if self._col - self._base < width:
                    break
                self._matched += 1
                self._base += width
                continue
            if c == ">":
                self._matched += 1
                self._col += 1
                self._base = self._col
                self._gt = True
                return False
            break
        if self._fence:
            if self._matched == len(self._containers):
                self._phase, self._leaf = "leaf", "fence"
                self._closing = self._col - self._base <= 3
                return True
            self._end_fence()  # 所在的容器结束了，围栏随之结束
        if self._code:
            if self._matched == len(self._containers) and self._col - self._base >= 4:
                self._phase, self._leaf = "leaf", "code"
                return True
            self._code = False
        self._phase = "open"
        return True

    def _open(self, c: str) -> bool:
        """开新的容器；返回 `True` 表示这个字符是这一行的内容（已进入 leaf 阶段）。"""
        if c in " \t":
            self._advance(c)
            return False
        self._gt = False
        rel = self._col - self._base
        if rel >= 4 and self._lazy_rel() <= 3 and c in ">`~#-*_":
            self._rules.append((self._matched, "", 0))
            return self._start_maybe(c, rel)
        if rel <= 3 and c == ">":
            self._rules.clear()  # 块引用之前开始的部分不会是分隔线
            self._close_unmatched()
            self._push(_Container("quote", 0))
            self._matched = len(self._containers)
            self._col += 1
            self._base = self._col
            self._gt = True
            self._new_block()
            return False
        if rel <= 3 and (c in "-+*" or "0" <= c <= "9"):
            self._marker = "bullet" if c in "-+*" else "digits"
            self._marker_col = self._col
            self._number, self._digits = (0, 0) if c in "-+*" else (int(c), 1)
            self._col += 1
            self._rules.append((self._matched, "", 0))
            self._shape(c)
            return False
        if rel <= 3:
            self._rules.append((self._matched, "", 0))
        return self._start_leaf(c, rel)

    def _lazy_rel(self) -> int:
        """这一行可能是没匹配上的列表项里段落的续行时，相对最里层那个列表项的内容的缩进（可以是负数）；不是这种情况
        （没有真的段落、全都匹配上了、没匹配上的容器里有块引用）时返回一个大数。块引用的懒惰续行按外层的缩进判断（同
        markdown-it）。"""
        depth = len(self._containers)
        if (
            not (self._para and self._settled)
            or self._matched == depth
            or (self._quotes and self._quotes[-1] >= self._matched)
        ):
            return _BIG
        return self._col - self._base - (self._widths[depth] - self._widths[self._matched])

    def _marker_char(self, c: str) -> bool:
        """列表标记读到一半；返回 `True` 表示这个字符是这一行的内容。"""
        match self._marker:
            case "digits":
                if "0" <= c <= "9" and self._digits < 9:
                    self._number = self._number * 10 + int(c)
                    self._digits += 1
                    self._col += 1
                    self._shape(c)
                    return False
                if c in ".)":
                    self._marker = "delim"
                    self._col += 1
                    self._shape(c)
                    return False
                return self._not_marker()
            case "bullet" | "delim":
                if c not in " \t":
                    return self._not_marker()
                self._marker, self._marker_end = "spaces", self._col
                self._advance(c)
                return False
            case _:  # spaces
                if c in " \t":
                    self._advance(c)
                    return False
                if not self._open_item(empty=False):
                    return self._not_marker()
                return self._open(c)

    def _not_marker(self) -> bool:
        """以为是列表标记、其实不是：从标记的第一个字符起是这一行的内容（那几个字符里没有反引号、`!` 与 `[`）。"""
        self._marker = ""
        self._start_leaf("-", self._marker_col - self._base)
        return True

    def _open_item(self, *, empty: bool) -> bool:
        """标记之后有内容（或到了行尾）：开列表项。在同一层容器里打断段落的列表项不能是空的，有序的要从 1 开始（外层
        容器没匹配上、像是懒惰续行时不受这个限制，同 markdown-it）；不是列表项时返回 `False`。"""
        if empty and self._marker in ("bullet", "delim"):
            self._marker_end = self._col
        same_container = self._matched == len(self._containers)
        if (
            self._para
            and same_container
            and (empty or (self._marker_end - self._marker_col > 1 and self._number != 1))
        ):
            return False
        spaces = self._col - self._marker_end
        base = self._marker_end + (spaces if 1 <= spaces <= 4 and not empty else 1)
        self._close_unmatched()
        self._push(_Container("item", base - self._base))
        self._matched = len(self._containers)
        self._base = base
        self._marker = ""
        self._new_block()
        return True

    def _start_leaf(self, c: str, rel: int) -> bool:
        self._phase = "leaf"
        self._shape_ok = rel <= 3
        unmatched = self._matched < len(self._containers)
        following = self._definition_next
        if following == "title" and c not in "\"'(":
            self._definition_next = following = ""
        if following:
            # 链接引用定义的后续行（像段落的续行一样，没匹配上的容器留着）。缩进不到 4 列、像围栏或 ATX 标题的，
            # 同时看它是不是（是的话定义到此结束，这一行按那种块算）。
            self._leaf = "definition"
            self._definition = {"dest": "ws1", "title": "ws2", "more": "title"}[following]
            self._closer = self._definition_closer
            self._lazy = unmatched
            self._def_code = self._def_more = True
            if rel <= 3 and c in "`~":
                self._def_opener, self._mark, self._mark_col = True, c, self._col
            elif rel <= 3 and c == "#":
                self._def_atx = True
            return True
        if rel >= 4:
            if self._para and self._settled:
                self._leaf, self._lazy = "text", unmatched  # 缩进代码不能打断段落：是续行
            else:
                self._close_unmatched()
                self._new_block()
                self._leaf, self._code = "code", True
            return True
        if c == "<":
            self._risky = True
        elif c == "$":
            self._dollar = 1
        if c in "`~":
            self._leaf, self._mark, self._mark_col = "opener", c, self._col
        elif c == "#":
            self._leaf = "atx"
            return True  # 是标题还是段落，看下一个字符
        elif c == "[" and (not self._para or self._after_definition):
            self._leaf, self._definition = "definition", "label"
            self._close_unmatched()
            self._new_block()
            return True
        else:
            self._leaf = "text"
        self._continue_or_start()
        return True

    def _continue_or_start(self) -> None:
        """这一行是段落的文字：接着开着的段落（没匹配上的容器暂且当作懒惰续行留着），或者开新的段落。"""
        if self._para and self._settled:
            self._lazy = self._matched < len(self._containers)
        else:
            self._close_unmatched()
            self._new_block()

    def _start_maybe(self, c: str, rel: int) -> bool:
        """列表项里段落的续行，相对列表项的内容缩进不到 4 列、以块开头的字符起头：是块的开头的话列表项结束，这一行是
        外层的缩进代码（它在外层缩进了 4 列以上）；不是的话是懒惰续行。到确定为止按代码算，行内代码照常配对。"""
        self._phase, self._leaf = "leaf", "maybe"
        self._shape_ok = True
        _ = rel
        if c == ">":
            self._maybe_block = True
            self._resolve_maybe()
            return True
        if c in "`~":
            self._maybe = "run"
        elif c == "#":
            self._maybe = "hash"
        else:
            self._maybe = "rule"  # 只可能是分隔线（列表标记在这里不打断段落），到行尾看
        self._mark = c
        return True

    def _maybe_char(self, c: str) -> None:
        count = self._maybe_count
        match self._maybe:
            case "run":
                if not self._after and c == self._mark:
                    self._maybe_count += 1
                    return
                if not self._after:
                    self._after = True
                    if count < 3:
                        self._maybe_block = False
                if self._mark == "`" and c == "`":
                    self._maybe_block = False
            case "hash":
                if c == "#" and count < 7:
                    self._maybe_count += 1
                    return
                self._maybe_block = c in " \t" and 1 <= count <= 6
            case (
                _
            ):  # rule：分隔线的可能都没了（`_shape()` 已读过这个字符）就是懒惰续行，否则到行尾看
                if c in " \t" or self._rules:
                    return
                self._maybe_block = False
        if self._maybe_block is not None:
            self._resolve_maybe()

    def _resolve_maybe(self) -> None:
        if self._maybe_block:
            self._lazy = False
            self._close_unmatched()
            self._new_block()
            self._leaf, self._code = "code", True
        else:
            self._leaf, self._lazy = "text", True

    def _content(self, c: str) -> _Fence:
        if c not in " \t":
            self._shape(c)
        if self._dollar == 1:
            self._dollar = 2
        elif self._dollar == 2:
            self._risky = self._risky or c == "$"
            self._dollar = 0
        match self._leaf:
            case "fence":
                self._closing_char(c)
                return self._fence_where()
            case "code":
                return _Fence.CODE
            case "maybe":
                where = self._span_char(c)
                self._maybe_char(c)
                # 在这个字符上定成了懒惰续行：它已是段落的文字。
                return where if self._maybe_block is False else _Fence.CODE
            case "opener":
                self._opener_char(c)
                where = self._span_char(c)
                if self._after and (self._marks < 3 or (self._mark == "`" and self._tick_in_info)):
                    self._leaf = "text"  # 不可能是围栏的开头了：是段落，行内代码照常配对
                    return where
                return _Fence.CODE
            case "atx":
                if c == "#" and self._hashes < 7:
                    self._hashes += 1
                    return _Fence.NONE
                if c in " \t" and 1 <= self._hashes <= 6:
                    self._heading()
                    return _Fence.NONE
                self._leaf = "text"  # 不是标题，是段落
                self._continue_or_start()
                return self._span_char(c)
            case "definition":
                if self._def_atx:
                    if c == "#" and self._hashes < 7:
                        self._hashes += 1
                    else:
                        self._def_atx = False
                        if c in " \t" and 1 <= self._hashes <= 6:
                            self._heading()  # ATX 标题：定义到此结束
                            return _Fence.NONE
                if self._def_opener:
                    self._opener_char(c)
                    if self._after and (
                        self._marks < 3 or (self._mark == "`" and self._tick_in_info)
                    ):
                        self._def_opener = False  # 不可能是围栏的开头行了
                return self._definition_char(c)
            case _:
                return self._span_char(c)

    def _heading(self) -> None:
        self._lazy = False
        self._close_unmatched()
        self._new_block()
        self._leaf = "heading"

    def _closing_char(self, c: str) -> None:
        if not self._closing:
            return
        if self._after:
            self._closing = c in " \t"
        elif self._marks:
            if c == self._fence:
                self._marks += 1
            elif c in " \t" and self._marks >= self._fence_length:
                self._after = True
            else:
                self._closing = False
        elif c == self._fence:
            self._marks = 1
        else:
            self._closing = False

    def _opener_char(self, c: str) -> None:
        if self._after:
            if c == "`":
                self._tick_in_info = True
        elif c == self._mark:
            self._marks += 1
        else:
            self._after = True
            if c == "`":
                self._tick_in_info = True

    def _definition_char(self, c: str) -> _Fence:
        """链接引用定义：标签照常（行内代码照常配对），冒号之后按代码算并检查地址与标题。"""
        where = self._span_char(c)
        state = self._definition
        if state == "label":
            self._label += 1
            if self._label == 1:  # 开头的 `[`
                return where
            # 标签不限长度：CommonMark 规定方括号里最多 999 个字符，参照的 markdown-it-py 不限；按定义算（标题里的
            # 图片写法不认）只会少认，所以取不限的一边。
            if c == "]" and not self._def_escape:
                self._definition = "colon" if self._label_text else "bad"  # 标签不能只有空白
            elif self._def_escape:
                self._def_escape = False
                self._label_text = True
            elif c == "\\":
                self._def_escape = True
            elif c == "[":
                self._definition = "bad"
            elif c not in " \t":
                self._label_text = True
            return where
        if state == "colon":
            self._definition = "ws1" if c == ":" else "bad"
            self._def_code = c == ":"
            return _Fence.CODE if c == ":" else where
        if state != "bad":
            self._definition_step(c)
            if self._definition == "bad" and not self._def_opener:
                # 确定不是定义（或这一行不是它的标题）：从这个字符起是正文（之前标成代码的不回头改，只会少认）。
                # 同时可能是围栏开头行的不算：那样它的信息串仍是代码。
                self._def_code = False
        return _Fence.CODE if self._def_code else where

    def _definition_step(self, c: str) -> None:
        state = self._definition
        if self._def_escape:
            self._def_escape = False
            return
        match state:
            case "ws1":
                if c in " \t":
                    return
                if c == "<":
                    self._definition = "angle"
                else:
                    self._definition = "bare"
                    self._definition_step(c)
            case "angle":
                if c == "\\":
                    self._def_escape = True
                elif c == ">":
                    self._definition = "after"
                elif c == "<":
                    self._definition = "bad"
            case "bare":
                if c == "\\":
                    self._def_escape = True
                elif c == "(":
                    self._parens += 1
                    if self._parens > MAX_URL_PARENS:
                        self._definition = "bad"
                elif c == ")":
                    if self._parens == 0:
                        self._definition = "bad"
                    else:
                        self._parens -= 1
                elif c in " \t":
                    self._definition = "bad" if self._parens else "ws2"
            case "after":
                self._definition = "ws2" if c in " \t" else "bad"
            case "ws2":
                if c in " \t":
                    return
                if c in "\"'(":
                    self._definition, self._closer = "title", ")" if c == "(" else c
                else:
                    self._definition = "bad"
            case "title":
                if c == "\\":
                    self._def_escape = True
                elif c == self._closer:
                    self._definition = "after_title"
                elif self._closer == ")" and c == "(":
                    self._definition = "bad"
            case "after_title":
                if c not in " \t":
                    self._definition = "bad"
            case _:
                pass

    def _span_char(self, c: str) -> _Fence:
        if c == "`" and not (self._escaped and not self._span):
            self._run += 1
            self._escaped = False
            return _Fence.CODE
        self._end_run()
        if not self._span:
            self._escaped = c == "\\" and not self._escaped
            return _Fence.NONE
        return _Fence.CODE

    def _end_run(self) -> None:
        if self._run:
            if not self._span:
                self._span = self._run
            elif self._span == self._run:
                self._span = 0
            self._run = 0

    def _shape(self, c: str) -> None:
        """这一行（新开的容器之后）是不是分隔线、setext 标题的下划线。
        分隔线的各项读到这个字符之后，还留着的都以它为标记；早开始的一项读过的字符更多（个数不少于后面的），之后的变化
        又都一样，所以只留最早的一项。"""
        survivor = next(
            ((index, count) for index, mark, count in self._rules if mark in ("", c)), None
        )
        self._rules = (
            [] if survivor is None or c not in "-*_" else [(survivor[0], c, survivor[1] + 1)]
        )
        if self._underline != "!":
            continued = c in "=-" and self._underline in ("", c) and not self._underline_gap
            self._underline = c if continued else "!"

    # ---- 行尾 ----

    def _end_line(self) -> None:
        self._end_run()
        self._escaped = False  # 行尾的反斜杠不转义下一行的字符
        empty_item = False
        if self._phase == "open" and self._marker:
            if self._marker == "digits" or self._thematic_line() or not self._open_item(empty=True):
                # 只有数字，或打断不了段落的空列表项：是段落的内容；整行是分隔线（如 `- - -`）：按分隔线算
                self._not_marker()
            else:
                empty_item = True
        if self._phase == "leaf":
            self._end_content()
            self._empty_item = -1
        elif empty_item:
            # 只有标记的一行：空的列表项。紧接着空行的话它就结束了（列表项开头最多一个空行）。
            self._new_block()
            self._empty_item = len(self._containers)
        else:
            self._end_blank()
        if self._risky:
            self._off = True
        self._new_line()

    def _end_blank(self) -> None:
        # 空行：列表项都算匹配上，少了 `>` 的块引用关掉（其中的围栏、缩进代码随之结束）；还空着的列表项结束。
        quote = bisect.bisect_left(self._quotes, self._matched)  # 第一个没匹配上的块引用
        matched = self._quotes[quote] if quote < len(self._quotes) else len(self._containers)
        if self._empty_item == len(self._containers) == matched:
            matched -= 1
        self._empty_item = -1
        self._definition_next = ""
        self._after_definition = False
        if matched < len(self._containers):
            if self._fence:
                self._end_fence()
            self._truncate(matched)
            self._code = False
        self._para = False
        self._settled = True
        self._clear_spans()

    def _end_content(self) -> None:
        self._after_definition = False
        if self._leaf != "definition":
            self._definition_next = ""
        match self._leaf:
            case "fence":
                if self._closing and (self._after or self._marks >= self._fence_length):
                    self._end_fence()
                return
            case "code" if not self._thematic_line():  # 列表标记之后缩进 4 列以上的分隔线仍是分隔线
                return
            case "maybe":
                # 行尾还没定：围栏的开头（够长、反引号围栏的信息串里没有反引号）、只有 `#` 的标题、分隔线是块的开头；
                # 其余是懒惰续行。
                ended = self._maybe == "hash" and 1 <= self._maybe_count <= 6
                fence = (
                    self._maybe == "run" and self._maybe_count >= 3 and self._maybe_block is None
                )
                self._maybe_block = ended or fence or self._thematic_line()
                self._resolve_maybe()
                if self._maybe_block:
                    return
            case "heading":
                self._para = False
                self._clear_spans()
                return
            case "atx" if 1 <= self._hashes <= 6:  # 只有 `#` 的标题
                self._heading()
                self._para = False
                return
            case "opener" if self._marks >= 3 and not (self._mark == "`" and self._tick_in_info):
                self._open_fence()
                return
            case "definition" if self._def_more:
                if (
                    self._def_opener
                    and self._marks >= 3
                    and not (self._mark == "`" and self._tick_in_info)
                ):
                    self._open_fence()  # 围栏的开头：定义到此结束
                    return
                if self._def_atx and 1 <= self._hashes <= 6:
                    self._heading()  # 只有 `#` 的标题
                    self._para = False
                    return
                underline = self._underline not in ("", "!") and self._shape_ok
                if self._thematic_line() or (underline and self._definition_next != "title"):
                    if self._thematic_line():
                        self._undo_rule_containers()
                    self._lazy = False  # 分隔线或 setext 标题的下划线：定义到此结束
                    self._close_unmatched()
                    self._new_block()
                    return
                self._end_definition()
                return
            case "definition":
                self._end_definition()
                return
            case _:
                pass
        thematic = self._thematic_line()
        underline = (
            self._underline not in ("", "!") and self._shape_ok and self._para and not self._lazy
        )
        if thematic or underline:
            # 分隔线或 setext 标题的下划线：段落到此结束；分隔线撤回它之后这一行开的容器。
            if thematic:
                self._undo_rule_containers()
            self._lazy = False
            self._close_unmatched()
            self._new_block()
            return
        if self._lazy:
            return  # 懒惰续行：容器留着，段落接着
        self._close_unmatched()
        self._para, self._settled = True, True

    def _open_fence(self) -> None:
        self._lazy = False
        self._close_unmatched()
        self._new_block()
        self._fence, self._fence_length = self._mark, self._marks
        self._fence_top = (
            not self._containers
            and self._mark == "`"
            and self._mark_col == 0
            and not self._off
            and not self._risky
        )

    def _thematic_line(self) -> bool:
        return any(count >= 3 for _, _, count in self._rules)

    def _undo_rule_containers(self) -> None:
        """这一行是分隔线：撤回从最外层的那个分隔线起这一行开的容器（它们的标记其实是分隔线的一部分）。"""
        start = next(index for index, _, count in self._rules if count >= 3)
        self._truncate(start)
        self._matched = min(self._matched, len(self._containers))

    def _end_definition(self) -> None:
        state = self._definition
        complete = state in ("after", "ws2", "after_title") or (
            state == "bare" and not self._parens
        )
        if complete:
            # 完整的定义（标题可能还在下一行）：不是段落，行内代码清掉。
            self._lazy = False
            self._close_unmatched()
            self._new_block()
            self._definition_next = "" if state == "after_title" else "title"
            self._after_definition = True
            return
        if state in ("ws1", "title"):
            # 地址还没出现、标题还没完：要看下一行。拿不准它真是定义，按段落接着配对行内代码，但它不是能懒惰续行的段落。
            self._definition_next = "dest" if state == "ws1" else "more"
            self._definition_closer = self._closer
            self._after_definition = True
            if not self._lazy:
                self._close_unmatched()
            self._para, self._settled = True, False
            return
        # 不是（或不再是）定义：是段落的文字（接着前面拿不准的段落，或者是新的段落）。
        self._definition_next = ""
        if not self._lazy:
            self._close_unmatched()
        self._para, self._settled = True, True

    # ---- 小工具 ----

    def _fence_where(self) -> _Fence:
        return _Fence.TOP if self._fence_top else _Fence.CODE

    def _end_fence(self) -> None:
        self._fence, self._fence_length, self._fence_top = "", 0, False
        self._new_block()

    def _close_unmatched(self) -> None:
        if self._matched < len(self._containers) and not self._lazy:
            self._truncate(self._matched)
            self._new_block()

    def _push(self, container: _Container) -> None:
        if len(self._containers) >= MAX_CONTAINERS:
            self._overflow = True  # 太深了：之后都按代码算（见 `mark()`），容器不再增加
            return
        if container.kind == "quote":
            self._quotes.append(len(self._containers))
        self._containers.append(container)
        self._widths.append(self._widths[-1] + container.width)

    def _truncate(self, depth: int) -> None:
        """只留容器栈的前 `depth` 层。"""
        del self._containers[depth:]
        del self._quotes[bisect.bisect_left(self._quotes, depth) :]
        del self._widths[depth + 1 :]

    def _new_block(self) -> None:
        self._para = False
        self._settled = True
        self._code = False
        self._definition_next = ""
        self._after_definition = False
        self._clear_spans()

    def _clear_spans(self) -> None:
        self._span = self._run = 0
        self._escaped = False


class _Piece(Enum):
    PROSE = "prose"  # 认图片，认不出的 `![` 换掉
    VERBATIM = "verbatim"  # 顶格的反引号围栏代码块的内容：原样放行
    CODE = "code"  # 其余的代码：不认图片，`![` 一律换掉


class ImageScanner:
    """一条回复用一个实例。`feed()` 收模型的原始文字增量，先经内部的 `MarkdownGuard`，再从中认出图片；输出是
    文字与 `ImageRef` 交替的列表（相邻的文字已合并，没有空串）。全部输入完后调用 `finish()`。"""

    def __init__(self, start_bytes: int = 0) -> None:
        """`start_bytes`：同一条流式消息里，这个扫描器的输出之前已经有多少字节（UTF-8）。它不为零时整条回答都不原样
        放行：字节数说明不了前面的文字停在什么 Markdown 状态（例如前面是 `x` 时，这里顶格的 `` ```md `` 在最终的消息里
        其实是 `` x```md ``，不是围栏），而闸门只看这个扫描器收到的原文。字节照常计数。"""
        self._guard = MarkdownGuard()
        self._blocks = _BlockTracker()  # 正文还是代码（识别得准不准）
        self._gate = _Gate(off=start_bytes > 0)  # 能不能原样输出（安全）
        # 还不能确定的正文：guard 的输出与对应的原文（等长），从某个 `!` 或 `\` 开始。
        self._out: str = ""
        self._src: str = ""
        # 代码或原样放行的内容的最后一个字符是 `!`：要看下一个字符是不是 `[`。`_bang_verbatim` 表示它在字节上限之内、
        # 是原样放行的内容，下一个字符也是原样放行的 `[` 时可以留着。
        self._bang = False
        self._bang_verbatim = False
        # 被代码截断的链接地址：之后还有这么多个字符可能仍是它的一部分，其中的 `![` 只换掉、不认（拿不准时不认）。
        # `_tail_escape` 表示其中最后一个正文字符是起转义作用的反斜杠。
        self._tail = 0
        self._tail_escape = False
        # 图片语法读到地址或标题里时遇到的代码段：先扣住，等它有了结论（见 `_resolve_held()`）。各项是 `_piece()` 的参数。
        self._held: list[tuple[_Piece, str, str, str]] = []
        # 看过的原文（正文与代码）的行尾状态，见 `_lines()`：这一行到目前为止只有空白（前面是行尾），上一个字符是
        # `\r`。用于空行结束 `_tail`、清掉 `_brackets`。
        self._line_blank = True
        self._line_cr = False
        # 正文里没被转义、还没配上的 `[`（含 `![` 的）有几个：`]` 没有可配的 `[` 时它后面的 `(…)` 不是链接的后半段。
        # 只会多算（多算只让之后的 `](` 更像链接、少认图片）：链接里不能再有链接这条规则没有跟踪。
        self._brackets = 0
        self._written = start_bytes  # 已经输出了多少字节（每张图按 `IMAGE_DISPLAY_BYTES` 算）

    @property
    def guard(self) -> MarkdownGuard:
        """内部的 guard，用于 `closing_fence()`、`fence_opening`。"""
        return self._guard

    def feed(self, text: str) -> list[ScanItem]:
        """`text` 里孤立的 Unicode 代理码位先去掉（同 `llm.clean_text()`，适配器已经去过一遍，这里兜底）：它们编码
        不成 UTF-8，数不了字节。"""
        if not text.isascii():
            text = "".join(c for c in text if not 0xD800 <= ord(c) <= 0xDFFF)
        return self._take(self._guard.feed_segments(text), final=False)

    def finish(self) -> list[ScanItem]:
        return self._take(self._guard.finish_segments(), final=True)

    def _take(self, segments: Sequence[Segment], *, final: bool) -> list[ScanItem]:
        out = _Output(self._written)
        for segment in segments:
            assert len(segment.text) == len(segment.source), "guard 的输出与原文要等长"
            blocks = self._blocks.mark(segment.source)
            gate = self._gate.mark(segment.source)
            start = 0
            for (where, released), run in itertools.groupby(zip(blocks, gate, strict=True)):
                end = start + len(list(run))
                # 原样放行要三方都同意：闸门（`_Gate`）说能放行，块跟踪器（`_BlockTracker`）说是顶格的反引号围栏的
                # 内容，guard 也当作反引号围栏（分条时靠 guard 补上与重开围栏）；少一个就按代码处理。其余位置是正文
                # 还是代码只看块跟踪器。
                source = segment.source[start:end]
                matching = source
                if where is _Fence.TOP and released and segment.kind is SegmentKind.FENCE:
                    piece = _Piece.VERBATIM
                elif where is _Fence.NONE:
                    piece = _Piece.PROSE
                elif where is _Fence.PREFIX:
                    # 行首的容器前缀（缩进与 `>`）：是正文，但跨行的链接语法里不算字符，认语法时当作空格。`!`、`[`、
                    # `]`、`\` 照样留着：块跟踪器不会把它们算作前缀，万一算了也不能让 `![` 不经扫描就输出。
                    piece = _Piece.PROSE
                    matching = "".join(c if c in "![]\\" else " " for c in source)
                else:
                    piece = _Piece.CODE
                self._piece(out, piece, segment.text[start:end], source, matching)
                start = end
        if final:
            while self._held:
                # 重放扣住的各段时，后面的图片候选可能又扣住了一些段：一直处理到没有扣住的为止，不然它们就丢了。
                self._resolve_held(out, final=True)
            self._settle_bang(out, "")
            self._scan(out, final=True)
        self._written = out.written
        return out.result()

    def _piece(self, out: _Output, piece: _Piece, text: str, source: str, matching: str) -> None:
        """`matching` 是认语法时用的原文（容器前缀换成了空格），与 `source` 等长。"""
        self._settle_bang(out, source, verbatim=piece is _Piece.VERBATIM)
        if self._held or (piece is _Piece.CODE and self._in_image_target()):
            # 图片语法已经读到 `](` 之后（地址或标题里）：这里的反引号只是地址或标题的字符（同 CommonMark，链接的后半段
            # 先于行内代码解析），先不截断，扣住这些段，等图片语法有了结论再说（见 `_resolve_held()`）。
            self._held.append((piece, text, source, matching))
            self._resolve_held(out, final=False)
            return
        if piece is not _Piece.PROSE:
            # 代码截断了没写完的语法。先截断再算 `_tail`：截断它的这一段也计入预算，结论才不随切分变化。截断时扣住的
            # `!` 紧接着按这一段的第一个字符定下来，免得它挪到这一段后面。
            self._scan(out, final=True, cut=True)
            self._settle_bang(out, source)
            if self._tail:
                self._spend_tail(out, piece, text, source, matching)  # 也记下这一段的行尾
            else:
                self._lines(source, stop=False)
            if piece is _Piece.VERBATIM:
                self._verbatim(out, source)
            else:
                self._defuse(out, text)
            return
        if self._tail:
            text, source, matching = self._spend_tail(out, piece, text, source, matching)
            if not text:
                return
        self._out += text
        self._src += matching
        self._scan(out, final=False)

    def _in_image_target(self) -> bool:
        """攒着的正文以一个还没有结论的图片语法开头，而且它的说明已经结束、读到了 `](` 之后。"""
        src = self._src
        if not src.startswith("![") or not _alt_closed(src):
            return False
        return _match(src[: MAX_SYNTAX_CHARS + 1], 0, final=False) is _Verdict.MORE

    def _resolve_held(self, out: _Output, *, final: bool) -> None:
        """攒着的正文（以图片语法开头）接上扣住的各段，看图片语法有没有结论：还要更多输入就接着扣；是图片就交出它，
        扣住的各段去掉它用掉的部分之后照常处理；不是图片就当作被第一段代码截断（同没扣住时），各段照常处理。
        `_match()` 的结论只取决于读到的字符，所以与切分无关；最多扣 `MAX_SYNTAX_CHARS` 个字符，且不跨行。"""
        if not self._held:
            return
        combined = self._src + "".join(matching for _, _, _, matching in self._held)
        window = combined[: MAX_SYNTAX_CHARS + 1]
        result = _match(window, 0, final=final or len(window) < len(combined))
        if result is _Verdict.MORE:
            return
        held, self._held = self._held, []
        if result is _Verdict.NO:
            self._scan(out, final=True, cut=True)
        else:
            image, length = result
            self._lines(combined[:length], stop=False)
            out.image(image)
            used = length - len(self._src)
            self._src = self._out = ""
            trimmed: list[tuple[_Piece, str, str, str]] = []
            last = _Piece.PROSE  # 图片语法最后一个字符（`)`）所在那一段的分类
            for piece, text, source, matching in held:
                cut = min(used, len(text))
                used -= cut
                if cut:
                    last = piece
                if cut < len(text):
                    trimmed.append((piece, text[cut:], source[cut:], matching[cut:]))
            held = trimmed
            if last is not _Piece.PROSE:
                # 标题里有不成对的反引号：块跟踪器仍以为在行内代码里，之后的分类正好反了。到这一段结束（下一个
                # 空行）都不再认图片，`![` 一律换掉（借用 `_tail` 的做法），只会少认。
                self._tail, self._tail_escape = _BIG, False
        for piece, text, source, matching in held:
            self._piece(out, piece, text, source, matching)

    def _spend_tail(
        self, out: _Output, piece: _Piece, text: str, source: str, matching: str
    ) -> tuple[str, str, str]:
        """这一段开头的字符计入 `_tail`；其中的正文不认图片，照样换掉 `![`。返回剩下的正文；代码与原样放行的内容只计数，
        原样返回。预算的最后一个正文字符是起转义作用的反斜杠时，被它转义的那个字符也算在内（`\\!` 不是图片的开头）。
        遇到空行时预算到此用完：链接的地址与标题都不能跨过空行。"""
        if piece is not _Piece.PROSE:
            used = min(len(text), self._tail)
            blank = self._lines(source[:used], stop=True)
            self._tail = 0 if blank >= 0 else self._tail - used
            self._lines(source[used if blank < 0 else blank :], stop=False)
            self._tail_escape = False
            return text, source, matching
        while self._tail and text:
            used = min(len(text), self._tail)
            # 其中的 `[` 也算上（多算只会让之后的 `](` 更像链接、少认图片），空行之前的随空行清掉。
            blank = self._lines(matching[:used], stop=True, brackets=True)
            if blank >= 0:
                used, self._tail = blank, 0
            else:
                self._tail -= used
            head = matching[:used]
            slashes = len(head) - len(head.rstrip("\\"))
            carried = self._tail_escape and slashes == len(head)
            self._tail_escape = carried != (slashes % 2 == 1)
            self._defuse(out, text[:used])
            text, source, matching = text[used:], source[used:], matching[used:]
            if text:
                self._settle_bang(out, source)
            if not self._tail and self._tail_escape:
                self._tail = 1
        return text, source, matching

    def _lines(self, text: str, *, stop: bool, brackets: bool = False) -> int:
        """接着看原文里的这些字符（行尾是 `\\n`、`\\r\\n` 或单独的 `\\r`，容器前缀已换成空格），记下行尾的状态；每个
        字符只看一次，逐个字符地记着状态，所以结论不随切分变化。遇到空行时没配上的方括号都清掉（`_brackets`：方括号
        不能跨段落配对）；`stop` 时在那里停下，返回结束空行的那个行尾之后的位置。没停下时返回 -1。`brackets` 时其中的
        `[` 也计入 `_brackets`，与空行的清零按先后在同一遍里处理（成批处理与逐个字符处理的结果相同）。"""
        start = 0
        for match in _LINE_END.finditer(text):
            run = text[start : match.start()]
            self._line_run(run)
            if brackets:
                self._brackets += run.count("[")
            start = match.end()
            c = match.group()
            if self._line_cr and c == "\n":
                self._line_cr = False  # `\r\n` 的后半，与前面的 `\r` 同属一个行尾
                continue
            blank = self._line_blank
            self._line_blank, self._line_cr = True, c == "\r"
            if blank:
                self._brackets = 0
                if stop:
                    return start
        self._line_run(text[start:])
        if brackets:
            self._brackets += text.count("[", start)
        return -1

    def _line_run(self, run: str) -> None:
        if run:
            self._line_cr = False
            if run.strip(" \t"):
                self._line_blank = False

    def _defuse(self, out: _Output, text: str) -> None:
        """不认图片：后面跟着 `[` 或不可见字符的 `!` 一律换成「！」（`_defuses()`）；最后一个字符是 `!` 时要看下一个
        字符。"""
        body = _defuse_bangs(text)
        self._bang, self._bang_verbatim = body.endswith("!"), False
        out.text(body.removesuffix("!") if self._bang else body)

    def _verbatim(self, out: _Output, text: str) -> None:
        """原样放行：只有前面的字节数（`out.written` 加上这一段在它之前的部分）不超过 `VERBATIM_LIMIT_BYTES - 2` 的
        `![` 原样留着，其余的感叹号照样换掉（同 `_defuse()`）；最后一个字符是 `!` 时要看下一个字符。"""
        budget = VERBATIM_LIMIT_BYTES - 2 - out.written
        # `text[:kept]` 里的每个位置之前都不超过 `budget` 个字节：在那里开始的 `![` 原样留着。
        kept = len(text.encode()[:budget].decode(errors="ignore")) + 1 if budget >= 0 else 0
        kept = min(kept, len(text))
        body = text[:kept] + _defuse_bangs(text[kept:])
        self._bang = body.endswith("!")
        self._bang_verbatim = self._bang and len(text) <= kept
        out.text(body.removesuffix("!") if self._bang else body)

    def _settle_bang(self, out: _Output, following: str, *, verbatim: bool = False) -> None:
        """扣住的 `!` 后面接的是 `following`：是 `[` 或不可见字符（`_defuses()`）就换成「！」，除非两个都是原样放行的
        内容、`!` 又在字节上限之内（`verbatim` 表示 `following` 是原样放行的内容）。"""
        if self._bang:
            keep = not _defuses(following) or (self._bang_verbatim and verbatim)
            out.text("!" if keep else DEFUSED_BANG)
            self._bang = self._bang_verbatim = False

    def _scan(self, out: _Output, *, final: bool, cut: bool = False) -> None:
        """扫描攒着的正文：在原文上认图片，不是图片的部分输出 guard 改过的文字。`cut` 表示后面接着的是代码：没写完的
        图片语法到此不是图片；没写完的链接地址与标题拿不准，按地址处理。"""
        src, text = self._src, self._out
        i = tracked = 0  # `src[:tracked]` 已经记进行尾的状态（`_lines()`）
        while i < len(src):
            special = _SPECIAL.search(src, i)
            if special is None:
                out.text(text[i:])
                i = len(src)
                break
            k = special.start()
            out.text(text[i:k])
            self._lines(src[tracked:k], stop=False)  # 在这里下结论之前，前面的空行先清掉方括号
            tracked = k
            if src[k] == "[":
                self._brackets += 1
                out.text(text[k])
                i = k + 1
                continue
            if src[k] == "\\":
                if k + 1 == len(src):
                    if not final:
                        i = k  # 还不知道转义的是什么
                        break
                    out.text("\\")
                    i = k + 1
                    continue
                if src[k + 1] == "!":
                    # 转义的 `!` 不是图片的开头，但后面紧跟 `[` 时也换成全角：平台认不认这个转义没有实测。
                    if k + 2 == len(src) and not final:
                        i = k
                        break
                    if k + 2 == len(src) and cut:
                        # 正文到此为止，后面接着的是别的分类：`!` 扣住，按下一段的第一个字符定（同代码段结尾的 `!`）。
                        out.text("\\")
                        self._bang, self._bang_verbatim = True, False
                    else:
                        out.text("\\" + (DEFUSED_BANG if _defuses(src[k + 2 : k + 3]) else "!"))
                    i = k + 2
                    continue
                out.text(text[k : k + 2])
                i = k + 2
                continue
            if src[k] == "]":
                # 链接（或没认成的图片）的 `](地址 "标题")` 部分：里面的图片写法只是地址或标题的文字，不认图片，
                # `![` 一律换掉。链接的文字（方括号里）照常扫描。`]` 没有可配的 `[`（如开头的被转义了）时它只是字符。
                if not self._brackets:
                    out.text(text[k])
                    i = k + 1
                    continue
                if k + 1 == len(src) and not final:
                    i = k  # 还不知道后面是不是 `(`
                    break
                if src[k + 1 : k + 2] != "(":
                    self._brackets -= 1
                    out.text(text[k])
                    i = k + 1
                    continue
                window = src[k + 1 : k + 1 + MAX_SYNTAX_CHARS + 1]
                truncated = k + 1 + len(window) < len(src)
                tail = _link_tail(window, final=truncated or (final and not cut))
                if tail is _Verdict.MORE and cut:
                    self._defuse(out, text[k:])  # 结尾的 `!` 扣住，看截断它的代码的第一个字符
                    self._tail = max(0, MAX_SYNTAX_CHARS - (len(src) - k))
                    self._tail_escape = False
                    # 拿不准它是不是链接：里面的 `[` 也算上（多算只会少认）。`_link_tail()` 给出 MORE 时这里没有空行。
                    self._brackets -= 1
                    self._lines(src[k:], stop=False, brackets=True)
                    i = tracked = len(src)
                    break
                if tail is _Verdict.MORE:
                    i = k
                    break
                self._brackets -= 1
                if tail is _Verdict.NO:
                    out.text(text[k])
                    i = k + 1
                    continue
                end = k + 1 + tail
                out.text(_defuse_bangs(text[k:end]))  # 以 `)` 结尾，没有扣住的 `!`
                i = end
                continue
            # 只把至多 `MAX_SYNTAX_CHARS + 1` 个字符交给 `_match()`：更长的不可能是图片，这样每个候选的开销有上限，
            # 一段文字里有很多个候选时整体仍是线性的。
            window = src[k : k + MAX_SYNTAX_CHARS + 1]
            result = _match(window, 0, final=final or k + len(window) < len(src))
            if result is _Verdict.MORE:
                i = k
                break
            if result is _Verdict.NO:
                # 「不是图片」只在看到 `!` 之后的字符（或已到全文结尾）时给出，这里的判断不随切分变化。
                if k + 1 == len(src) and cut:
                    # 正文以这个 `!` 结尾、后面接着的是别的分类（代码或原样放行）：不能直接输出，扣住，按下一段的
                    # 第一个字符定（同代码段结尾的 `!`）。不靠块跟踪器前后两个字符的分类一致。
                    self._bang, self._bang_verbatim = True, False
                else:
                    out.text(DEFUSED_BANG if _defuses(src[k + 1 : k + 2]) else "!")
                i = k + 1
                continue
            image, length = result
            out.image(image)
            i = k + length
        self._lines(src[tracked:i], stop=False)
        self._src, self._out = src[i:], text[i:]


class _Output:
    """收集输出：文字先放进列表，遇到图片或取结果时才拼起来，所以总开销与输出的长度成正比。`written` 是到目前为止
    输出了多少字节（从 `written` 的初值算起；文字按 UTF-8，每张图按 `IMAGE_DISPLAY_BYTES`）。"""

    def __init__(self, written: int = 0) -> None:
        self.items: list[ScanItem] = []
        self._parts: list[str] = []
        self.written = written

    def text(self, text: str) -> None:
        if text:
            self._parts.append(text)
            self.written += len(text.encode())

    def image(self, image: ImageRef) -> None:
        self._flush()
        self.items.append(image)
        self.written += IMAGE_DISPLAY_BYTES

    def result(self) -> list[ScanItem]:
        self._flush()
        return self.items

    def _flush(self) -> None:
        if self._parts:
            self.items.append("".join(self._parts))
            self._parts = []


def _match(s: str, start: int, *, final: bool) -> tuple[ImageRef, int] | _Verdict:
    """从 `s[start]`（一个 `!`）开始的是不是图片语法：是就返回（图片，结束位置）。

    只按读到的字符下结论：`NO` 只在已读到的部分违反规则时给出，所以有了更多输入也不会变；读到结尾还没写完时给
    `MORE`（`final` 时给 `NO`）。整段的长度上限在两种情况下都检查，结论同样不随切分变化。
    """
    n = len(s)
    unfinished = _Verdict.NO if final or n - start > MAX_SYNTAX_CHARS else _Verdict.MORE
    i = start + 1
    if i == n:
        return unfinished
    if s[i] != "[":
        return _Verdict.NO
    i += 1
    alt_start = i
    depth = 0
    while True:
        if i - alt_start > MAX_ALT_CHARS:
            return _Verdict.NO
        if i == n:
            return unfinished
        c = s[i]
        # 没转义的 `|`：在表格里会被拆进两个单元格（见模块 docstring 的已知的限制）。
        if c in "\r\n|":
            return _Verdict.NO
        if c == "\\":
            if i + 1 == n:
                return unfinished
            if s[i + 1] in "\r\n":
                return _Verdict.NO
            i += 2
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            if depth == 0:
                break
            depth -= 1
        i += 1
    alt = _unescape(s[alt_start:i])[0]
    i += 1
    if i == n:
        return unfinished
    if s[i] != "(":
        return _Verdict.NO
    i = _skip_blanks(s, i + 1)
    if i == n:
        return unfinished
    angle = s[i] == "<"
    url_start = i + angle
    stops = _ANGLE_STOP if angle else _URL_STOP
    limit = min(n, url_start + MAX_URL_CHARS + 1)
    i = url_start
    parens = 0
    while True:
        stop = stops.search(s, i, limit)
        if stop is None:  # 读到上限或结尾都没有结束
            if limit - url_start > MAX_URL_CHARS or not _scheme_viable(s[url_start:limit]):
                return _Verdict.NO
            return unfinished
        i = stop.start()
        c = s[i]
        if c == "\\":
            # 反斜杠加 ASCII 标点是这个字符本身（被转义的括号不参与配平，`\>` 不结束尖括号地址）；别的反斜杠是
            # 普通字符，后面的字符照常看。
            if i + 1 == limit:  # 还不知道转义的是什么
                if limit - url_start > MAX_URL_CHARS or not _scheme_viable(s[url_start:limit]):
                    return _Verdict.NO
                return unfinished
            i += 2 if s[i + 1] in _PUNCTUATION else 1
            continue
        if c == "|":
            return _Verdict.NO
        if angle:
            if c != ">":  # 没转义的 `<`、换行
                return _Verdict.NO
            break
        if c in " \t":
            break
        if c == "(":
            parens += 1
            if parens > MAX_URL_PARENS:
                return _Verdict.NO
        elif c == ")":
            if parens == 0:
                break
            parens -= 1
        else:  # 换行与其它控制字符
            return _Verdict.NO
        i += 1
    url = s[url_start:i]
    if parens or not _scheme_viable(url):  # 裸地址里没转义的圆括号要配平
        return _Verdict.NO
    i += angle
    # 尖括号地址里可以有字面的空格；别的空白与控制字符不算图片。
    if angle and _DECODED_BAD.search(url.replace(" ", "")):
        return _Verdict.NO
    # 地址完整之后先在原文上处理转义与字符引用，再判断协议（白名单由调用方用解码之后的地址判断）；最后才把字面的
    # 空格写成 `%20`（同参照解析器）。先写的话，`\ ` 里的反斜杠会与拼出来的 `%` 被当成转义。
    decoded, bad = _unescape(url)
    if bad or not decoded.lower().startswith(_URL_SCHEMES):
        return _Verdict.NO
    url = decoded.replace(" ", "%20")
    blank = i
    i = _skip_blanks(s, i)
    if i == n:
        return unfinished
    if s[i] in "\"'(" and i > blank:
        opener = s[i]
        closer = ")" if opener == "(" else opener
        i += 1
        title_start = i
        while True:
            if i - title_start > MAX_TITLE_CHARS:
                return _Verdict.NO
            if i == n:
                return unfinished
            c = s[i]
            if c == closer:
                break
            if c in "\r\n|" or (opener == "(" and c == "("):
                return _Verdict.NO
            if c == "\\":
                if i + 1 == n:
                    return unfinished
                if s[i + 1] in "\r\n":
                    return _Verdict.NO
                i += 2
                continue
            i += 1
        i = _skip_blanks(s, i + 1)
        if i == n:
            return unfinished
    if s[i] != ")":
        return _Verdict.NO
    i += 1
    if i - start > MAX_SYNTAX_CHARS:
        return _Verdict.NO
    return ImageRef(alt=alt, url=url, raw=s[start:i]), i


def _alt_closed(s: str) -> bool:
    """`s` 以 `![` 开头：说明（按 `_match()` 的规则）已经在 `MAX_ALT_CHARS` 之内结束，后面紧跟着 `(`。"""
    i, depth = 2, 0
    while i < len(s) and i - 2 <= MAX_ALT_CHARS:
        c = s[i]
        if c in "\r\n":
            return False
        if c == "\\":
            i += 2
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            if depth == 0:
                return s[i + 1 : i + 2] == "("
            depth -= 1
        i += 1
    return False


def _link_tail(s: str, *, final: bool) -> int | _Verdict:
    """`s` 以 `(` 开头：是不是 CommonMark 链接的 `(地址 "标题")` 部分，是就返回它的长度。比图片的规则宽：地址可以
    是任何协议，前后的空白里可以有一个换行，标题可以是 `"…"`、`'…'` 或 `(…)`、可以跨行（不能有空行）。拿不准时
    宁可当作是（里面的图片写法就不认了）。"""
    unfinished = _Verdict.NO if final or len(s) > MAX_SYNTAX_CHARS else _Verdict.MORE
    i = _skip_space(s, 1)
    if i is None:
        return _Verdict.NO
    if i == len(s):
        return unfinished
    if s[i] == "<":
        i += 1
        while True:
            if i >= len(s):
                return unfinished
            c = s[i]
            if c == "\\":
                i += _escape_length(s, i)
                continue
            if c in "<\r\n":
                return _Verdict.NO
            i += 1
            if c == ">":
                break
    else:
        parens = 0
        while True:
            if i >= len(s):
                return unfinished
            c = s[i]
            if c == "\\":
                i += _escape_length(s, i)
                continue
            if c == "(":
                parens += 1
                if parens > MAX_URL_PARENS:
                    return _Verdict.NO
            elif c == ")":
                if parens == 0:
                    break
                parens -= 1
            elif c <= " " or c == "\x7f":
                break
            i += 1
        if parens:
            return _Verdict.NO
    blank = i
    after = _skip_space(s, i)
    if after is None:
        return _Verdict.NO
    i = after
    if i == len(s):
        return unfinished
    if s[i] == ")":
        return i + 1
    if i == blank or s[i] not in "\"'(":
        return _Verdict.NO
    opener = s[i]
    closer = ")" if opener == "(" else opener
    i += 1
    line_blank = False  # 标题里到目前为止的这一行只有空白（前面是换行）
    while True:
        if i >= len(s):
            return unfinished
        c = s[i]
        if c == "\\":
            i += _escape_length(s, i)
            line_blank = False
            continue
        if c == closer:
            break
        if opener == "(" and c == "(":
            return _Verdict.NO
        if c in "\r\n":
            if line_blank:
                return _Verdict.NO  # 空行
            if c == "\r" and s[i + 1 : i + 2] == "\n":
                i += 1
            line_blank = True
        elif c not in " \t":
            line_blank = False
        i += 1
    after = _skip_space(s, i + 1)
    if after is None:
        return _Verdict.NO
    if after == len(s):
        return unfinished
    return after + 1 if s[after] == ")" else _Verdict.NO


def _escape_length(s: str, i: int) -> int:
    """`s[i]` 是反斜杠：要跳过几个字符。反斜杠只转义 ASCII 标点（连同它跳过 2 个）；后面是别的字符（包括行尾）时
    反斜杠是普通字符，只跳过它自己，后面的字符照常看（行尾照常参与空行的判断）。后面还没有字符时也只跳过它自己，
    调用方读到结尾时按「还没写完」处理。"""
    return 2 if s[i + 1 : i + 2] in _PUNCTUATION else 1


def _skip_space(s: str, i: int) -> int | None:
    """跳过空格、制表符与至多一个行尾（\n、\r\n 或单独的 \r）；遇到空行返回 `None`。结尾的单独 \r 可能是 \r\n 的
    前半，调用方读到结尾时自会按「还没写完」处理。"""
    newline = False
    while i < len(s):
        c = s[i]
        if c in " \t":
            i += 1
        elif c in "\r\n":
            if newline:
                return None
            newline = True
            i += 2 if s[i : i + 2] == "\r\n" else 1
        else:
            break
    return i


def _skip_blanks(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t":
        i += 1
    return i


def _unescape(text: str) -> tuple[str, bool]:
    """按 CommonMark 处理反斜杠转义与字符引用（`&名字;`、`&#十进制;`、`&#x十六进制;`），同参照的解析器：反斜杠加一个
    ASCII 标点是这个字符本身，别的反斜杠原样保留；被转义的 `&` 不是字符引用的开头。不用 `html.unescape()`：它把不带
    分号的（如查询串里的 `&para=1`）也解码，会改坏正常的地址。认不出的名字原样保留；码位为 0、是代理码位或超出
    Unicode 范围的换成 U+FFFD。

    返回（结果，字符引用有没有解码出空白或控制字符）：地址里有这种的不算图片（原文里尖括号地址中的空格不受影响）。"""
    if "&" not in text and "\\" not in text:
        return text, False
    bad = False

    def decode(match: re.Match[str]) -> str:
        nonlocal bad
        escaped, decimal, hexadecimal, name = match.groups()
        if escaped is not None:
            return escaped
        if name is not None:
            decoded = html5.get(name + ";", match[0])
        else:
            code = int(decimal) if decimal is not None else int(hexadecimal, 16)
            invalid = code == 0 or 0xD800 <= code <= 0xDFFF or code > 0x10FFFF
            decoded = "\N{REPLACEMENT CHARACTER}" if invalid else chr(code)
        bad = bad or _DECODED_BAD.search(decoded) is not None
        return decoded

    return _ESCAPE_OR_REFERENCE.sub(decode, text), bad


def _scheme_viable(url: str) -> bool:
    """地址（或写了一半的地址）以 http(s):// 开头，或还可能是。前 8 个字符里有 `&` 或 `\\` 时，只看它之前的部分：
    之后的要解码才知道（`https&colon;//` 也是 https 的地址）。只看前缀，所以对一个前缀说「不可能」，对它的任何延长也
    都是「不可能」，结论不随切分变化。"""
    head = url[: len(_URL_SCHEMES[1])]
    cut = min((p for p in (head.find("&"), head.find("\\")) if p >= 0), default=-1)
    if cut >= 0:
        head = head[:cut].lower()
        return any(scheme.startswith(head) for scheme in _URL_SCHEMES)
    head = head.lower()
    return any(scheme.startswith(head) or head.startswith(scheme) for scheme in _URL_SCHEMES)


# ---- 改写 ----

MAX_SHOWN_ALT_CHARS = 60
"""图片语法与说明文字里的说明最多这么多个字符，超过的截断并加省略号。"""
DEFAULT_ALT = "图片"
"""说明为空时图片语法里写的文字（平台的图片语法里，说明是显示不出图时给的文字）。"""

# 去掉方括号、反斜杠、反引号；尖括号换成全角，免得说明文字经由我们的兜底文字或图片语法带进 HTML 标签；`&` 换成
# 全角，免得字符引用（如 `&num;`）在图片语法里被解码成别的字符（如带进宽高的 `#`）；`$`、`~` 换成全角，免得拼出
# `$$`、`~~~`（见 `clean_alt()`）；`|` 换成全角，免得在表格里把一格拆成两格。
_ALT_TRANSLATION = str.maketrans(
    {
        "[": None,
        "]": None,
        "\\": None,
        "`": None,
        "<": "＜",
        ">": "＞",
        "&": "＆",
        "$": "＄",
        "~": "～",
        "|": "｜",
    }
)
_SIZE_HINT = re.compile(r"#(?=\s*\d)")  # 平台从 `#数字px` 读宽高


def clean_alt(alt: str) -> str:
    """清理说明文字，使它放进图片语法后改不了语法的结构，也带不进别的宽高：换行与控制字符换成空格并合并空白，
    去掉方括号、反斜杠、反引号，尖括号、`&`、`$`、`~`、`|` 换成全角的「＜」「＞」「＆」「＄」「～」「｜」（`|` 会在
    表格里把一格拆成两格），紧跟数字的 `#` 换成全角的「＃」，过长的截断（最多 `MAX_SHOWN_ALT_CHARS` 个字符）。可能
    返回空串。

    结果里没有原样放行的闸门（`_Gate`）拿不准的任何标记，原文里就有的、删掉字符之后拼出来的都没有：闸门只看模型的
    原文，看不到我们换上去的文字。逐条对照：连续的反引号（反引号删掉了）、`~~~` 与 `$$`（`~`、`$` 换成全角）、
    `\\[` 这类（反斜杠删掉了）、行首的 `<`（尖括号换成全角）、单独的 `\\r`（换成空格）；`---`、`+++` 与 `<` 的规则
    只看行首，而说明文字在兜底文字与图片语法里都不在行首。"""
    text = "".join(" " if c.isspace() or not c.isprintable() else c for c in alt)
    text = _SIZE_HINT.sub("＃", text.translate(_ALT_TRANSLATION))
    text = " ".join(text.split())
    if len(text) > MAX_SHOWN_ALT_CHARS:
        text = text[: MAX_SHOWN_ALT_CHARS - 1].rstrip() + "…"
    return text


def render_image(alt: str, url: str, width: int, height: int) -> str:
    """平台要的图片语法 `![说明 #宽px #高px](地址)`。`url` 是素材库给的地址，`width`、`height` 是显示尺寸
    （`display_size()`）。说明经 `clean_alt()` 清理，为空时用 `DEFAULT_ALT`。

    Raises:
        ValueError: 宽高不是正整数，地址为空、含空白、控制字符、圆括号或尖括号（放进语法会改变它的结构）、反斜杠、
            反引号、`$$` 或 `~~~`（原样放行的闸门拿不准的标记，见 `clean_alt()`）、竖线（会拆开表格的单元格），或
            结果超过 `IMAGE_DISPLAY_BYTES` 个字节（扫描器按这个数计算原样放行的字节上限）。
    """
    if width < 1 or height < 1:
        raise ValueError("宽高要是正整数")
    if (
        not url
        or any(c.isspace() or not c.isprintable() or c in "()<>\\`|" for c in url)
        or "$$" in url
        or "~~~" in url
    ):
        raise ValueError("地址不能放进图片语法")
    rendered = f"![{clean_alt(alt) or DEFAULT_ALT} #{width}px #{height}px]({url})"
    if len(rendered.encode()) > IMAGE_DISPLAY_BYTES:
        raise ValueError("图片语法太长（地址太长）")
    return rendered


def fallback_text(alt: str) -> str:
    """去掉一张图（名单外、下载或上传失败）时留下的文字：「［图片：说明］」，说明清理后为空时是「［图片］」。方括号是
    全角的，和后面的 `(` 拼不成链接。从不返回空串：图片前面的 `!` 与后面的 `[...](...)` 会直接拼成一段新的图片语法。
    说明最多 `MAX_SHOWN_ALT_CHARS` 个字符，所以结果不超过 `IMAGE_DISPLAY_BYTES` 个字节。"""
    shown = clean_alt(alt)
    return f"［图片：{shown}］" if shown else "［图片］"


# 下载一张图的结果（下载器在 `image_download.py`）：对话流程（`conversation.py`）要用到这几个类型，又不该为此依赖
# httpx2，所以放在这里。


class Failure(StrEnum):
    """一张图没能下载的原因（固定的类别，不含地址）。"""

    REFUSED = "主机不在白名单里"
    REDIRECTS = "重定向次数过多"
    BAD_REDIRECT = "重定向地址不合法"
    HTTP = "HTTP 错误"
    ENCODED = "响应做了内容编码"
    NETWORK = "网络错误"
    TOO_LARGE = "超过大小上限"
    NOT_IMAGE = "不是支持的图片"
    TIMEOUT = "超时"


@dataclass(frozen=True, slots=True)
class FetchedImage:
    """下载到的一张图：内容、按文件头判断的 MIME 类型与原图的宽高（像素）。"""

    data: bytes = field(repr=False)
    mime_type: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class FetchFailed:
    """没能下载。`str()` 是记日志用的描述，只有类别、HTTP 状态码、异常类型与主机名。

    Attributes:
        reason: 类别。
        host: 出错的那一跳的主机名；不是合法的域名、或解析不出时为 `None`。
        status: HTTP 状态码（`Failure.HTTP` 时有）。
        error: 异常类型名（`Failure.NETWORK` 时有）。
    """

    reason: Failure
    host: str | None = None
    status: int | None = None
    error: str | None = None

    def __str__(self) -> str:
        details: list[str] = []
        if self.status is not None:
            details.append(f"HTTP {self.status}")
        if self.error is not None:
            details.append(self.error)
        if self.host is not None:
            details.append(f"主机 {self.host}")
        return self.reason.value + (f"（{'，'.join(details)}）" if details else "")
