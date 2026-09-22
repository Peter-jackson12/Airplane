"""Repository-only output boundaries; no original local data is accessed."""
from pathlib import Path
import subprocess

from notebooks.join_weather_full import write_new_json

ROOT = Path(__file__).resolve().parents[1]


def test_full_join_rows_are_ignored_but_new_json_is_trackable_and_lf():
    name = 'baseline_recovery_v2_contract_only'
    ignored = [f'data/weather_probe/{name}_full_join/joined_latency0min.csv.gz',
               f'data/weather_probe/{name}_full_join/parts/0000_latency0.csv']
    for path in ignored:
        result = subprocess.run(['git', 'check-ignore', '--no-index', '--quiet', path], cwd=ROOT)
        assert result.returncode == 0, path
    for suffix in ['manifest', 'summary']:
        path = f'output/{name}_full_weather_join_{suffix}.json'
        result = subprocess.run(['git', 'check-ignore', '--no-index', '--quiet', path], cwd=ROOT)
        assert result.returncode == 1, path
        attributes = subprocess.check_output(['git', 'check-attr', 'eol', '--', path], cwd=ROOT, text=True)
        assert attributes.strip() == f'{path}: eol: lf'


def test_new_json_bytes_have_explicit_lf_on_every_platform(tmp_path):
    path = tmp_path / 'summary.json'
    write_new_json(path, {'name': 'synthetic', 'rows': 3, 'measured_latency': False})
    raw = path.read_bytes()
    assert b'\r' not in raw and raw.endswith(b'\n')
    assert not path.with_suffix('.json.tmp').exists()
