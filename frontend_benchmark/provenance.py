"""채점 결과 캐시를 '내용'으로 무효화하는 도구 (감사 C4 · C5 · C6 · C16).

mtime(수정 시각)만 비교하면 GT 교체(cp -p), 다른 GT 폴더, import 하는 모듈 변경, 옛 백업 복원을 놓친다.
그래서 캐시 키에 입력 파일의 sha1 · 경로 · 코드 digest · params 를 넣는다.

sha1 캐시   큰 파일(frontend_output.h5 최대 300MB)을 매번 해시하면 느리므로 (실제 경로, 크기, mtime_ns) → sha1 을
            <RUNS>/.cache/sha1.json 에 둔다. 같은 크기·mtime 이면 다시 읽지 않는다.
            해시한 시각과 mtime 이 RACY_NS(2초) 안이면 그 항목은 믿지 않고 다음에도 다시 해시해 확인한다
            (파일 시각 해상도 안에서 같은 크기로 다시 쓰인 경우 대비, git 의 'racy clean' 과 같은 이유).
            확인 결과가 같으면 기록을 건드리지 않는다 — 내용이 바뀐 항목이 있을 때만 파일을 다시 쓴다.
코드 digest 모듈 파일과, 그 파일이 import 하는 로컬 모듈(이 폴더 · meridian_benchmark 패키지)의 추이 폐포(transitive closure).
            파일 바이트가 아니라 docstring 을 뺀 AST(ast.dump)의 sha1 — 주석·docstring 만 고친 것은 다시 채점할 이유가 아니다.
"""
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

RACY_NS = 2_000_000_000


def canon(obj):
    """정렬된 JSON (키 계산용)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)


def sha1_text(s):
    return hashlib.sha1(s.encode()).hexdigest()


def key_of(obj):
    return sha1_text(canon(obj))


class Hasher:
    def __init__(self, cache_path):
        self.path = Path(cache_path)
        try:
            self.d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.d = {}
        self.dirty = False
        self.n_hashed = 0
        self.bytes_hashed = 0

    def file(self, p):
        p = Path(p)
        st = os.stat(p)
        rp = str(p.resolve())
        e = self.d.get(rp)
        same_stat = bool(e) and e[0] == st.st_size and e[1] == st.st_mtime_ns
        if same_stat and e[3] - st.st_mtime_ns > RACY_NS:
            return e[2]
        h = hashlib.sha1()
        with open(p, 'rb') as fh:
            for c in iter(lambda: fh.read(1 << 20), b''):
                h.update(c)
        self.n_hashed += 1; self.bytes_hashed += st.st_size
        digest = h.hexdigest()
        if same_stat and e[2] == digest:
            return digest                                          # racy 항목 재확인 — 같으면 기록을 건드리지 않는다
        self.d[rp] = [st.st_size, st.st_mtime_ns, digest, time.time_ns()]
        self.dirty = True
        return digest

    def files_digest(self, paths, root=None):
        """여러 파일 → (이름, sha1) 목록의 sha1. root 를 주면 상대 경로를 이름으로."""
        items = []
        for p in sorted(Path(x) for x in paths):
            items.append([str(p.relative_to(root)) if root else p.name, self.file(p)])
        return key_of(items)

    def save(self):
        if not self.dirty:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.d, separators=(',', ':')))
        os.replace(tmp, self.path)
        self.dirty = False
        return True


# ---------------------------------------------------------------- 코드 digest
_AST = {}


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], 'value', None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
    return tree


def ast_dump(path):
    """docstring 을 뺀 AST 덤프 (주석은 AST 에 없다)."""
    return ast.dump(_strip_docstrings(ast.parse(Path(path).read_text(), filename=str(path))))


def ast_digest(path):
    p = str(Path(path).resolve())
    st = os.stat(p)
    k = (p, st.st_size, st.st_mtime_ns)
    if k not in _AST:
        _AST[k] = sha1_text(ast_dump(p))
    return _AST[k]


_NAMES = {}


def _imported_names(path):
    st = os.stat(path)
    k = (str(path), st.st_size, st.st_mtime_ns)
    if k in _NAMES:
        return _NAMES[k]
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
            names += [f'{node.module}.{a.name}' for a in node.names]
    _NAMES[k] = names
    return names


def _resolve(name, roots):
    """점 이름 → 로컬 파일들 (패키지 __init__.py 포함). sys.modules 에 올라온 모듈은 실제 import 된 파일을 쓴다."""
    out = []
    parts = name.split('.')
    for i in range(1, len(parts) + 1):
        sub = '.'.join(parts[:i])
        m = sys.modules.get(sub)
        f = getattr(m, '__file__', None) if m else None
        cand = []
        if f and any(_under(f, r) for r in roots):
            cand = [Path(f)]
        elif not f:
            for r in roots:
                base = Path(r).joinpath(*parts[:i])
                if base.with_suffix('.py').is_file():
                    cand = [base.with_suffix('.py')]; break
                if (base / '__init__.py').is_file():
                    cand = [base / '__init__.py']; break
        out += cand
    return out


def _under(f, root):
    try:
        Path(f).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def code_files(entry_files, roots):
    """entry 파일들 + import 하는 로컬 모듈의 추이 폐포. roots 밖(표준·site-packages)은 뺀다."""
    todo = [Path(f).resolve() for f in entry_files]
    seen = set()
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        for n in _imported_names(f):
            for g in _resolve(n, roots):
                g = g.resolve()
                if g not in seen:
                    todo.append(g)
    return sorted(seen)


def code_digest(entry_files, roots):
    """→ (digest, {파일: ast sha1})"""
    files = code_files(entry_files, roots)
    per = {str(f): ast_digest(f) for f in files}
    return key_of(sorted((Path(f).name, d) for f, d in per.items())), per


# ---------------------------------------------------------------- 쓰기
def write_text_if_changed(path, text):
    path = Path(path)
    try:
        if path.read_text() == text:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)
    return True


def write_json_if_changed(path, obj, **kw):
    kw.setdefault('indent', 2); kw.setdefault('ensure_ascii', False)
    return write_text_if_changed(path, json.dumps(obj, **kw))
