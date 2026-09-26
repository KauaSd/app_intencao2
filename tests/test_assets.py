import pathlib
import re
import shutil
import subprocess

import pytest

WEB = pathlib.Path(__file__).parents[1] / "web"
JS = WEB / "js"
MODULES = ["api.js", "format.js", "intents.js", "chat.js", "app.js"]


def _all_assets():
    return list(WEB.rglob("*.js")) + list(WEB.rglob("*.html")) + list(WEB.rglob("*.css"))


def test_every_asset_exists():
    assert (WEB / "index.html").is_file()
    assert (WEB / "styles.css").is_file()
    for name in MODULES:
        assert (JS / name).is_file(), name


@pytest.mark.parametrize("name", MODULES)
def test_every_module_has_balanced_delimiters(name):
    source = (JS / name).read_text(encoding="utf-8")
    assert source.count("{") == source.count("}"), name
    assert source.count("(") == source.count(")"), name


@pytest.mark.parametrize("name", MODULES)
def test_every_module_parses(name):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    result = subprocess.run([node, "--check", str(JS / name)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("name", MODULES)
def test_no_module_uses_innerhtml(name):
    assert "innerHTML" not in (JS / name).read_text(encoding="utf-8"), name


def test_no_remote_reference_anywhere():
    for path in _all_assets():
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text, path
        assert "https://" not in text, path


def test_every_relative_import_resolves():
    for name in MODULES:
        source = (JS / name).read_text(encoding="utf-8")
        for target in re.findall(r"from\s+['\"](\./[^'\"]+)['\"]", source):
            assert (JS / target).is_file(), f"{name} -> {target}"


def test_format_js_stays_import_free():
    source = (JS / "format.js").read_text(encoding="utf-8")
    assert "import" not in source


def test_index_loads_every_module_as_a_module():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'type="module"' in html
    assert 'src="./js/app.js"' in html
    for name in MODULES:
        assert name in html, name


def test_index_has_the_expected_controls():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for element_id in ("message", "expected-intent", "samples",
                       "require-supported", "detect", "clear"):
        assert f'id="{element_id}"' in html, element_id


def test_index_has_every_panel():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for panel in ("result", "explanation", "tokens", "trace", "language",
                  "chat", "log"):
        assert f'id="{panel}"' in html, panel


def test_every_control_has_a_label():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for element_id in ("message", "expected-intent", "samples"):
        assert f'for="{element_id}"' in html, element_id


def test_the_stylesheet_defines_badges_focus_and_reduced_motion():
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    assert ".badge" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css


def test_format_helpers():
    body = (JS / "format.js").read_text(encoding="utf-8").replace("export ", "")
    namespace = {}
    exec(compile(body, "format.js", "exec"), namespace)
    assert namespace["formatConfidence"](0.7532) == "75%"
    assert namespace["formatConfidence"](0) == "0%"
    assert namespace["formatScore"]({"score": 3.5, "min_score": 3.0}) == "3.5 / 3.0"
    assert namespace["formatScore"]({"score": 0.0, "min_score": 3.0}) == "0.0 / 3.0"
    assert namespace["escapeText"]("<img onerror=x>") == "<img onerror=x>"