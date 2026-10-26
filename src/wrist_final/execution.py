"""Pre-access execution freeze and separate final-test output boundaries."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from .protocol import ROOT, PUBLIC, read, sha, identity, write_new, verify_frozen, agents_local_only

RESULTS = PUBLIC/'test_results'
PRIVATE = ROOT/'outputs/final_evaluation/phase5d'
RUN = PRIVATE/'study_01'
LOCK = RESULTS/'execution_freeze.json'
MODULES = ('execution.py', 'test_data.py', 'test_runner.py', 'test_analysis.py', 'test_verification.py', 'test_reports.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def sources():
    paths = [ROOT/'src/wrist_final'/p for p in MODULES]+[ROOT/'tests/test_heldout_execution.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in paths}


def locked():
    cfg = verify_frozen(); lock = read(LOCK)
    require(lock['sources'] == sources() and lock['matrix_sha256'] == sha(PUBLIC/'final_experiment_matrix.json'), 'Execution freeze changed')
    require(sha(RESULTS/'EXECUTION_NOTES.md') == lock['execution_notes_sha256'], 'Execution notes changed')
    require(not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', *lock['sources'], str(LOCK)], cwd=ROOT, text=True).strip(), 'Commit execution code before test access')
    require(not subprocess.check_output(['git', 'rev-list', 'origin/main..HEAD'], cwd=ROOT, text=True).strip(), 'Push execution freeze before test access')
    require(subprocess.check_output(['git', 'ls-files', 'src/wrist_final/test_runner.py'], cwd=ROOT, text=True).strip(), 'Execution code is not committed')
    return cfg, lock


def access_guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, Path)):
        return
    path = Path(args[0]).resolve()
    require(not path.name.startswith('.env'), 'Secret-file access is forbidden')
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    import os
    writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if writing and (path.is_relative_to(ROOT/'data') or path.is_relative_to(ROOT/'outputs') and not path.is_relative_to(PRIVATE)):
        raise PermissionError('Historical data/output writes are forbidden')


def historical_snapshot():
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    return {name: sha(ROOT/name) for name in names if not Path(name).name.startswith('.env')}


def prepare():
    from .protocol import install_phase5c_guard
    install_phase5c_guard()
    verify_frozen(); agents_local_only()
    require(not LOCK.exists(), 'Preserve the existing execution freeze')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    write_new(PRIVATE/'historical_before.json', historical_snapshot())
    # All pre-access tests use synthetic files and saved validation evidence only.
    temp = PRIVATE/'synthetic_tmp'; temp.mkdir(exist_ok=True); tempfile.tempdir = str(temp.resolve())
    began = time.perf_counter()
    with (PRIVATE/'pre_access_tests.log').open('x', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    require(result.wasSuccessful(), 'Pre-access tests failed; preserve the log')
    from .verify import validation_checks
    checked = validation_checks()
    lock = {'phase': '5D', 'authorized_request_sha256': '648345ba4ed62dd2d03a1ad9bbcbbb65dcc503ebbe91ec68c7c1858c48b8631e',
        'base_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources': sources(), 'matrix_sha256': sha(PUBLIC/'final_experiment_matrix.json'),
        'execution_notes_sha256': sha(RESULTS/'EXECUTION_NOTES.md'),
        'pre_access_tests': {'passed': True, 'count': result.testsRun, 'seconds': time.perf_counter()-began,
                             'private_log_sha256': sha(PRIVATE/'pre_access_tests.log')},
        'saved_validation_checks': checked, 'test_content_accessed_before_freeze': False,
        'condition_order': ['final_'+v for v in 'ABCDEF'], 'run': 'study_01',
        'full_passes_per_condition': 1, 'synthetic_warmups_per_condition': 3, 'no_test_pilot_or_replay': True}
    write_new(LOCK, lock)
    print(f'Pre-access freeze prepared: {result.testsRun} tests pass; commit and push before init', flush=True)


if __name__ == '__main__':
    prepare()
