"""Small CHIA workflow: supervisor -> worker -> isolated tests -> frozen full test.

Run: python loop.py --corpus /path/to/corpus.json --hours 6 --detach
The existing Codex login is used. Neither agent receives an earlier private port.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import io
import tarfile
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import uuid


def harness_identity(folder):
    files = {p.name: p.read_text() for p in Path(folder).iterdir()
             if p.is_file() and p.suffix in ('.py', '.md', '.json')}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frozen_environment(run, manifest):
    frozen = Path(run).resolve() / 'harness'
    if harness_identity(frozen) != manifest['harness_sha256']:
        raise ValueError('frozen harness identity mismatch')
    repository = Path(manifest['repository_root'])
    if not repository.is_absolute() or repository == Path('/'):
        raise ValueError('invalid repository isolation root')
    return {**os.environ, 'PYTHONPATH': str(frozen), 'NLLB_REPOSITORY_ROOT': str(repository)}


# Verify archived Python, instructions and model identities before importing any
# campaign modules in the re-executed controller. This is not a resume command.
if __name__ == '__main__' and '--controller' in sys.argv:
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument('--run-dir', required=True)
    checked, _ = check.parse_known_args()
    archive = Path(checked.run_dir).resolve()
    receipt = json.loads((archive / 'manifest.json').read_text())
    environment = frozen_environment(archive, receipt)
    if Path(__file__).resolve() != archive / 'harness/loop.py':
        raise ValueError('controller must execute its archived loop.py')
    os.environ.update({k: environment[k] for k in ('PYTHONPATH', 'NLLB_REPOSITORY_ROOT')})

import ray
from chia.base.ChiaFunction import ChiaFunction, get
from chia.base.tools.ChiaTool import ChiaTool
from agent import worker_node, supervisor_node, repository_root
from runner import safe_file, write_json, validate_log_request, full_job_timeout
from models import model_spec
WEIGHT_SHA256 = model_spec("600m")["files"]["pytorch_model.bin"]["sha256"]
from cluster import activated, identity as digest, load as load_cluster, ssh, upload, remote, reference_job

HERE = Path(__file__).resolve().parent
CORPUS_HASH = 'c9b03ca2525db184b6feaf341a52383ec1d9d74311682462c29ba036e74327d1'
PRECISIONS = ('bf16', 'fp32', 'fp16', 'bfp8_b')


def snapshot(workspace):
    files = {}
    for path in sorted(Path(workspace).rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks are not allowed')
        if path.is_file() and path.suffix in ('.py', '.md', '.json', '.cpp', '.hpp', '.h', '.txt'):
            if path.stat().st_size > 2_000_000:
                raise ValueError('file too large')
            files[str(path.relative_to(workspace))] = path.read_text()
    if sum(len(x.encode()) for x in files.values()) > 8_000_000:
        raise ValueError('workspace too large')
    return files


def code_hash(files):
    return digest({k: v for k, v in files.items() if not k.endswith('.md')})


def tool_name(run):
    """Use a stable bare TOML key; run names may contain dots or other punctuation."""
    return 'nllb_' + hashlib.sha256(str(Path(run).resolve()).encode()).hexdigest()[:16]



def suite_identity(run, model, stage):
    return digest({'model': model_spec(model), 'stage': stage, 'corpus': CORPUS_HASH,
                   'evaluator': (Path(run) / 'harness/evaluate.py').read_text()})


def passing_evaluation(result, job, signature, model, precision, stage):
    """Require an unambiguous trusted terminal receipt for this exact evaluation."""
    if not isinstance(result, dict):
        return False
    verdict = result.get('verdict')
    if not isinstance(verdict, dict) or not isinstance(verdict.get('precision'), dict):
        return False
    return (type(result.get('returncode')) is int and result['returncode'] == 0
            and not any(result.get(key) for key in ('failure', 'timeout', 'host_memory_limit'))
            and result.get('job') == job and result.get('cluster_sha256') == signature
            and result.get('model') == model and result.get('precision') == precision
            and verdict.get('passed') is True and not verdict.get('failures')
            and verdict.get('stage') == stage and verdict.get('model_key') == model
            and verdict['precision'].get('requested') == precision
            and isinstance(verdict.get('suite_sha256'), str) and bool(verdict['suite_sha256']))


def supervisor_facts(workspace, run, config, model="600m"):
    files = snapshot(workspace)
    saved_state = files.get('STATE.md', '')
    state_excerpt = saved_state
    if len(saved_state) > 12000:
        marker = '\n[Middle omitted; worker can read the full STATE.md.]\n'
        state_excerpt = saved_state[:6000] + marker + saved_state[-(6000 - len(marker)):]
    source, jobs, passed, candidates = code_hash(files), [], [], {}
    signature = digest(config)
    delivery_models = ('600m', '1.3b-distilled', '3.3b')
    evidence = {name: {precision: {stage: {'status': 'missing', 'job': None,
                                         'historical_jobs': []}
                                  for stage in ('smoke', 'bringup')}
                       for precision in ('bf16', 'bfp8_b')} for name in delivery_models}
    definitions = {(name, stage): suite_identity(run, name, stage)
                   for name in delivery_models for stage in ('smoke', 'bringup')}
    for path in sorted((Path(run) / 'jobs').glob('*/request.json'), key=lambda p: p.stat().st_mtime):
        request = json.loads(path.read_text())
        result_path = path.with_name('result.json')
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        failed = bool(result.get('returncode') or result.get('failure')
                      or result.get('timeout') or result.get('host_memory_limit'))
        background = request.get('kind') == 'reference-study'
        item = {'id': path.parent.name, 'kind': request.get('kind', 'evaluate'),
                'stage': request.get('stage', 'experiment'), 'model': request.get('model', '600m'),
                'cluster_sha256': request.get('cluster_sha256'),
                'background_nonblocking': background, 'pending': not result_path.exists(),
                'failure': result.get('failure'),
                'passed': False if failed else result.get('verdict', {}).get('passed')}
        jobs.append(item)
        precision, stage = request.get('precision'), item['stage']
        if (item['kind'] == 'evaluate' and item['model'] in evidence
                and precision in evidence[item['model']] and stage in ('smoke', 'bringup')):
            cell = evidence[item['model']][precision][stage]
            current = (request.get('code_sha256') == source
                       and request.get('cluster_sha256') == signature
                       and request.get('suite_definition_sha256') == definitions[item['model'], stage])
            if not current:
                cell['historical_jobs'] = (cell['historical_jobs'] + [item['id']])[-3:]
                if cell['job'] is None:
                    cell['status'] = 'historical'
            else:
                verdict = result.get('verdict', {})
                verified = passing_evaluation(result, item['id'], signature, item['model'], precision, stage)
                cell.update(job=item['id'], status=('pending' if item['pending'] else
                            'passed' if verified else 'failed' if failed
                            or verdict.get('passed') is False else 'unverified'))
        if (result and request.get('code_sha256') and request.get('stage') in ('smoke', 'bringup')
                and request.get('cluster_sha256') == digest(config)):
            group = candidates.setdefault((item['model'], request.get('precision')), {'sources': set(), 'passing': set()})
            group['sources'].add(request['code_sha256'])
            if item['passed']:
                group['passing'].add(request['code_sha256'])
        if (item['passed'] and request.get('code_sha256') == source
                and request.get('cluster_sha256') == digest(config)
                and not (item['model'] in evidence and precision in evidence[item['model']]
                         and stage in ('smoke', 'bringup'))):
            passed.append({'stage': item['stage'], 'model': item['model'], 'precision': request.get('precision')})
    # Keep the legacy list's shape, but use the same latest-current decisions as
    # the delivery matrix for its cells; an older pass cannot hide a later failure.
    passed.extend({'stage': stage, 'model': name, 'precision': precision}
                  for name, precisions in evidence.items() for precision, stages in precisions.items()
                  for stage, cell in stages.items() if cell['status'] == 'passed')
    baselines = [json.loads(path.read_text()) for path in (Path(run) / 'baselines').rglob('*.json')]
    baselines = [value for value in baselines if value.get('model') == model
                 and value.get('cluster_sha256') == digest(config)]
    return {'objective': 'Deliver ONE reusable NLLB source across 600M, distilled 1.3B and 3.3B; '
                         'the selected model is a development anchor, not the whole deliverable. '
                         'Prioritize missing shared-source model gates and the standalone artifact before '
                         'repeated selected-model-only profiling. Establish a baseline, profile it, '
                         'measure useful optimization hypotheses and relevant supported precision tradeoffs, '
                         'produce an importable configurable backend with checkpoint/input validation, standalone '
                         'text translation CLI and ordinary tests, then freeze for full validation. '
                         'A first bringup pass alone is not the optimization goal. Advance independent '
                         'implementation when runtime diagnosis blocks final evaluation.',
            'selected_model': model, 'baselines': baselines[-12:],
            'delivery_scope': {'models': list(delivery_models), 'development_anchor': model,
                'artifact': 'One configuration-driven source, standalone text translation CLI and portable tests.',
                'precision_policy': 'BF16 across all three models; investigate supported BFP8_B with honest '
                    'quality, latency and memory tradeoffs. Missing cells are not support or passes; '
                    'BFP8_B is block float, not IEEE FP8. Do not inherit CUDA precision results.',
                'evidence_scope': 'Latest matching-source/config/suite evaluation per cell, using trusted '
                    'job receipts only. Historical IDs are context, never current passes. Short gates '
                    'do not establish full quality, optimality or production readiness.',
                'short_gates': evidence},
            'worker_state': {'present': 'STATE.md' in files, 'text': state_excerpt,
                             'available_chars': len(saved_state), 'truncated': len(saved_state) > 12000,
                             'note': 'Worker-authored handoff, not verified evidence. Use job receipts for '
                                     'liveness and correctness; continue concrete work after a timeout.'},
            'environment': {'cluster_sha256': digest(config),
                            'tt_runtime': config['tt'].get('metal'),
                            'tt_python': config['tt'].get('python'),
                            'note': 'Jobs from a different cluster identity are historical. A runtime/machine '
                                    'switch requires revalidation, not repeating unchanged old-runtime diagnosis.'},
            'evaluated_candidates': [{'model': key[0], 'precision': key[1],
                'distinct_sources': len(value['sources']), 'passing_sources': len(value['passing'])}
                for key, value in candidates.items()], 'workspace': {'has_backend': 'backend.py' in files, 'source_sha256': source,
                          'source_sha256_scope': 'All non-Markdown workspace files; this is not backend.py alone.',
                          'backend_sha256': hashlib.sha256(files['backend.py'].encode()).hexdigest() if 'backend.py' in files else None,
                          'backend_sha256_scope': 'UTF-8 bytes of backend.py only; compare worker file hashes here.',
                          'file_names': sorted(files, key=lambda name: (name.endswith('.md'), name))[:64],
                          'implementation_files': sorted(name for name in files if not name.endswith('.md'))[:64],
                          'implementation_file_count': sum(not name.endswith('.md') for name in files),
                          'current_source_passes': passed},
            'recent_jobs': [j for j in jobs[:-12] if j['pending']] + jobs[-12:],
            'controls': {'run': {'public_input_stage': ['', 'smoke', 'bringup'],
                                  'timeout_seconds': 'bounded to 1..300 seconds',
                                  'model': list(delivery_models), 'default_model': model},
                         'log': 'Read own TT job stdout/stderr in byte pages up to 65536; no reference-study logs.',
                         'status': 'Use compact=True for bounded log tails; structured results remain complete.',
                         'evaluate': {'stage': ['smoke', 'bringup'], 'precision': list(PRECISIONS),
                                      'model': ['600m', '1.3b-distilled', '3.3b']},
                         'fixed_suites': {'smoke': {'cases': 1, 'max_new_tokens': 4},
                                          'bringup': {'cases': 8, 'max_new_tokens': 8}},
                         'unsupported': ['case selection/count', 'language selection', 'card selection',
                                         'evaluation timeout override'],
                         'full': 'Only submit(precision, model) after current-source bringup passes.'}}


def reference_progress(config, job):
    records = sorted(Path(job).glob('*-job.json'), key=lambda p: p.stat().st_mtime)
    if not records:
        return {'phase': 'preparing_reference', 'background_nonblocking': True}
    saved = json.loads(records[-1].read_text())
    endpoint = config['reference']
    if saved['endpoint_sha256'] != digest(endpoint):
        raise ValueError('reference progress endpoint mismatch')
    target = shlex.quote(saved['directory'] + '/output/progress.json')
    payload = ssh(endpoint, 'if test -f ' + target + '; then cat ' + target
                  + "; else echo '{\"phase\":\"reference_starting\"}'; fi", timeout=10)
    progress = json.loads(payload)
    return {'phase': progress.get('phase', 'reference_running'), 'background_nonblocking': True}


def seed_workspace(workspace, seed_run, model, seed_job=None):
    """Copy an explicitly selected fresh campaign, never its acceptance records."""
    root = Path(seed_run).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    seed_model = manifest.get('model')
    trusted = ('CONTRACT.md', 'TT_GUIDE.md', 'TASK.md')
    if (sorted(manifest.get('seed_files', [])) != sorted(trusted)
            and manifest.get('seed_policy') != 'fresh-public-nllb'):
        raise ValueError('seed must be a fresh public-source NLLB campaign')
    submission = root / 'submission.json'
    if seed_job:
        if Path(seed_job).name != seed_job or seed_job in ('.', '..'):
            raise ValueError('seed job must be a single job identifier')
        job = safe_file(root, 'jobs/' + seed_job)
        files = json.loads((job / 'source.json').read_text())
        request = json.loads((job / 'request.json').read_text())
        seed_model = request.get('model', seed_model)
        result = json.loads((job / 'result.json').read_text())
        if (request.get('kind') != 'evaluate' or request.get('stage') != 'bringup'
                or result.get('returncode') != 0 or result.get('timeout') or result.get('failure')
                or result.get('verdict', {}).get('passed') is not True
                or code_hash(files) != request.get('code_sha256')):
            raise ValueError('seed job must contain a passing, intact bringup source')
    elif submission.exists():
        frozen = json.loads(submission.read_text())
        seed_model = frozen.get('model', seed_model)
        files = frozen['files']
        if code_hash(files) != frozen['code_sha256']:
            raise ValueError('seed submission source identity mismatch')
    else:
        files = snapshot(Path(manifest['workspace']))
    omitted, retained = [], []
    for name, content in files.items():
        if Path(name).as_posix() in trusted:
            continue
        if Path(name).suffix == '.md':
            evidence = seed_model == model and Path(name).name.startswith(
                ('BASELINE_', 'BF16_BASELINE_', 'CURRENT_BF16_', 'CANDIDATE_', 'BFP8_'))
            if name not in ('RESEARCH.md', 'REPORT.md') and not evidence:
                omitted.append(name)
                continue
            retained.append(name)
        target = safe_file(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (Path(workspace) / 'STATE.md').write_text('# New campaign: ' + model + '\n\n'
        'Executable source and selected evidence were carried from an explicitly '
        'selected earlier fresh campaign. Old job IDs and results are historical: this campaign '
        'has no inherited pending jobs or passes. Follow the current selected-model TASK.md and '
        'CONTRACT.md, adapt the checkpoint, and re-evaluate. Retained baselines and evidence '
        'are historical under the archived runtime, not current instructions or passes. '
        'After a runtime change, revalidate the source and its same-precision baseline; then '
        'advance optimization and standalone deliverables. Do not repeat resolved original-runtime '
        'investigations unless new evidence warrants them.\n')
    return {'seed_run': str(root), 'seed_job': seed_job, 'seed_code_sha256': code_hash(files),
            'omitted_seed_notes': sorted(omitted), 'retained_seed_notes': sorted(retained)}


def prepare(run, config):
    """Verify a replacement before activating it; never reset a machine."""
    endpoint = config['tt']
    root = endpoint['work_dir'] + '/' + Path(run).name
    upload(endpoint, root + '/harness', {
        **{name: (Path(run) / 'harness' / name).read_bytes() for name in ('runner.py', 'evaluate.py', 'models.py', 'models.json')},
        'runtime.json': json.dumps(endpoint)})
    q = shlex.quote
    commit = ssh(endpoint, 'git -C ' + q(endpoint['metal']) + ' rev-parse HEAD', timeout=120).decode().strip()
    if commit != endpoint['metal_commit']:
        raise ValueError('replacement TT runtime/weight identity mismatch')
    weights = endpoint.get('weights_by_model', {'600m': endpoint.get('weights')})
    if not weights or any(not path for path in weights.values()):
        raise ValueError('configured model checkpoint paths are required')
    for model, location in weights.items():
        spec = model_spec(model)
        names = spec['weight_files'] + ([spec['weight_index']] if spec['weight_index'] else [])
        paths = [str(Path(location) / name) for name in names]
        command = 'sha256sum ' + ' '.join(q(p) for p in paths)
        if len(names) == 1:
            command = 'if test -f ' + q(location) + '; then sha256sum ' + q(location) + '; else ' + command + '; fi'
        actual = ssh(endpoint, command, timeout=600).decode().splitlines()
        if [line.split()[0] for line in actual] != [spec['files'][name]['sha256'] for name in names]:
            raise ValueError('replacement TT runtime/weight identity mismatch: ' + model)
        write_json(Path(run) / 'checkpoints' / (digest(config) + '-' + model + '.json'),
                   {'model': model, 'revision': spec['revision'], 'verified_weight_files': names})
    found = remote(config, run, {'kind': 'health'})['cards']
    for card in found:
        job = 'preflight-' + uuid.uuid4().hex
        result = remote(config, run, {'kind': 'preflight', 'job': job, 'pci': card['pci'],
                                      'model': next(iter(weights))}, timeout=150)
        write_json(Path(run) / 'preflights' / (job + '.json'), result)
    health = remote(config, run, {'kind': 'health'})
    if not any(c['state'] == 'healthy' for c in health['cards']):
        raise RuntimeError('replacement has no healthy TT cards')
    write_json(Path(run) / 'clusters' / (digest(config) + '.json'), config)
    write_json(Path(run) / 'cluster.json', config)
    return health


def refresh_cluster(run, path, current):
    replacement = load_cluster(path)
    if digest(replacement) == digest(current):
        return current, 'unchanged'
    pending = [p for p in (Path(run) / 'jobs').glob('*/request.json') if not p.with_name('result.json').exists()]
    if pending:
        return current, 'draining'
    prepare(run, replacement)
    return replacement, 'changed'


def recover_timed_out_card(run, config, health, model):
    """Try one real preflight per card/config after a recorded terminal timeout.

    Never reset hardware, recover memory-limit failures, or clear quarantine by
    editing health records. A saved submission intent prevents duplicate probes
    after transport ambiguity. Pending TT jobs must drain before this is called.
    """
    run = Path(run)
    requests = list((run / 'jobs').glob('*/request.json'))
    if any(not p.with_name('result.json').exists()
           and json.loads(p.read_text()).get('kind') != 'reference-study' for p in requests):
        return health
    signature = digest(config)
    for card in health['cards']:
        if card['state'] != 'quarantined':
            continue
        job = card.get('job', '')
        if not job or Path(job).name != job or job in ('.', '..'):
            continue
        folder = run / 'jobs' / job
        if not (folder / 'result.json').exists() or not (folder / 'request.json').exists():
            continue
        result = json.loads((folder / 'result.json').read_text())
        request = json.loads((folder / 'request.json').read_text())
        if (result.get('failure') != 'timeout' or not result.get('timeout')
                or result.get('host_memory_limit') or result.get('pci') != card['pci']
                or result.get('returncode') != 124 or result.get('job') != job
                or result.get('cluster_sha256') != signature
                or request.get('cluster_sha256') != signature):
            continue
        receipt = run / 'recoveries' / (digest([signature, card['pci']]) + '.json')
        if receipt.exists():
            continue
        probe = {'kind': 'preflight', 'job': 'preflight-' + uuid.uuid4().hex,
                 'pci': card['pci'], 'model': model}
        record = {'failed_job': job, 'request': probe, 'cluster_sha256': signature,
                  'state': 'submitted'}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        try:
            with receipt.open('x') as stream:
                json.dump(record, stream, indent=2)
        except FileExistsError:
            continue
        checked = remote(config, str(run), probe, timeout=170)
        write_json(run / 'preflights' / (probe['job'] + '.json'), checked)
        write_json(receipt, {**record, 'state': 'complete', 'result': checked})
        # The runner owns card state; success is established by its actual probe.
        return remote(config, str(run), {'kind': 'health'})
    return health




def validate_reference_archive(path, corpus, stage, model):
    """Inspect immutable short-stage oracle bytes before any remote extraction."""
    from evaluate import make_suite, suite_model, validate_config, validate_receipt, validate_artifact
    from models import unique_object
    if stage not in ('smoke', 'bringup'):
        raise ValueError('only completed short reference stages can be imported')
    path = Path(path)
    if not 0 < path.stat().st_size <= 512 * 1024 ** 2:
        raise ValueError('reference archive exceeds the short-stage size limit')
    data, entries = path.read_bytes(), {}
    required = {'input/suite.json', 'input/config.json', 'input/inputs.npz',
                'oracle/oracle.npz', 'oracle/reference.json', 'oracle/labels.json'}
    allowed = required | {'input/checkpoint_receipt.json'}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for entry in archive:
            name = entry.name.rstrip('/')
            if (name in entries or name not in allowed | {'input', 'oracle'}
                    or not (entry.isfile() or entry.isdir())
                    or entry.isdir() != (name in ('input', 'oracle'))):
                raise ValueError('unsafe reference archive entry')
            entries[name] = entry
        if not required <= entries.keys():
            raise ValueError('incomplete reference archive')
        def read(name):
            if entries[name].size > 2_000_000:
                raise ValueError('reference metadata too large')
            return json.loads(archive.extractfile(entries[name]).read(), object_pairs_hook=unique_object)
        spec, meta = read('input/suite.json'), read('oracle/reference.json')
        expected, _ = make_suite(corpus, stage, model)
        pinned, identity = suite_model(spec)
        if (identity['model_key'] != model or spec.get('model') != pinned['repo_id']
                or spec.get('revision') != pinned['revision']):
            raise ValueError('reference model identity missing or different')
        for key in ('stage', 'corpus_sha256', 'max_source_tokens', 'gates', 'timed_repeats'):
            if spec.get(key) != expected[key]:
                raise ValueError('reference corpus or fixed suite differs: ' + key)
        cases = [{k: v for k, v in case.items() if k != 'target_id'} for case in spec.get('cases', [])]
        if cases != expected['cases']:
            raise ValueError('reference cases differ from the frozen stage')
        if [case.get('name') for case in meta.get('cases', [])] != [case['name'] for case in spec['cases']]:
            raise ValueError('partial reference case results')
        policy = meta.get('precision', {})
        effective = policy.get('effective', {})
        if (policy.get('requested') != 'fp32' or meta.get('parameter_dtypes') != ['torch.float32']
                or any(effective.get(k) != 'fp32' for k in ('mode', 'weights', 'activations'))):
            raise ValueError('import requires an explicit FP32 reference')
        validate_config(read('input/config.json'), pinned)
        if spec['version'] == 1:
            weight = pinned['files'][pinned['weight_files'][0]]['sha256']
            if spec.get('weight_sha256') != weight or meta.get('weight_sha256') != weight:
                raise ValueError('legacy reference weight identity missing or different')
        else:
            if 'input/checkpoint_receipt.json' not in entries:
                raise ValueError('missing checkpoint identity receipt')
            validate_receipt(read('input/checkpoint_receipt.json'), identity, pinned, spec)
            validate_receipt(meta.get('checkpoint_receipt'), identity, pinned, spec)
            validate_artifact(meta, spec, identity)
    return data


def import_references(run, source, config, model):
    """Explicit verified data reuse; never import candidate code or acceptance."""
    source, run = Path(source).resolve(), Path(run)
    manifest = json.loads((source / 'manifest.json').read_text())
    old_hash = digest({p.name: p.read_text() for p in (source / 'harness').glob('*.py')})
    if manifest.get('harness_sha256') not in (old_hash, harness_identity(source / 'harness')):
        raise ValueError('source harness identity missing or changed')
    corpus = json.loads((run / 'corpus.json').read_text())
    if manifest.get('corpus_sha256') != digest(corpus):
        raise ValueError('source corpus identity missing or changed')
    key, validated = digest(config['reference'])[:16], []
    for stage in ('smoke', 'bringup'):
        origin = source / 'references' / key / model / stage
        if not origin.exists() and model == '600m':
            origin = source / 'references' / key / stage
        ready, archive = origin / 'ready.json', origin / 'reference.tar'
        if (not ready.is_file() or not archive.is_file()
                or json.loads(ready.read_text()).get('endpoint_sha256') != digest(config['reference'])):
            raise ValueError('source reference incomplete or endpoint identity differs')
        data = validate_reference_archive(archive, corpus, stage, model)
        validated.append((stage, archive, data))
    for stage, origin, data in validated:
        cache = run / 'references' / key / model / stage
        gpu = config['reference']['work_dir'] + '/' + run.name + '/references/' + key + '/' + model + '/' + stage
        tt = config['tt']['work_dir'] + '/' + run.name + '/references/' + model + '/' + stage
        upload(config['reference'], gpu, {name: (run / 'harness' / name).read_bytes()
                                         for name in ('evaluate.py', 'models.py', 'models.json')})
        for endpoint, directory in ((config['reference'], gpu), (config['tt'], tt)):
            ssh(endpoint, 'mkdir -p ' + shlex.quote(directory) + ' && tar -xf - -C ' + shlex.quote(directory), data, 300)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / 'reference.tar').write_bytes(data)
        write_json(cache / 'import.json', {'source_archive': str(origin), 'source_run': str(source),
            'source_harness_sha256': manifest['harness_sha256'], 'archive_sha256': hashlib.sha256(data).hexdigest(),
            'scope': 'FP32 reference data only; no candidate passes inherited'})
        write_json(cache / (digest(config['tt']) + '.deployed.json'), {'tt_dir': tt})
        write_json(cache / 'ready.json', {'reference_dir': gpu, 'tt_dir': tt, 'model': model,
            'stage': stage, 'endpoint_sha256': digest(config['reference'])})

def cached_reference(run, stage, config, model):
    model_spec(model)
    endpoint, run = config['reference'], Path(run)
    key = digest(endpoint)[:16]
    cache = run / 'references' / key / model / stage
    ready, deployed = cache / 'ready.json', cache / (digest(config['tt']) + '.deployed.json')
    if not all(p.is_file() for p in (ready, deployed, cache / 'reference.tar')):
        return None
    expected = {'reference_dir': endpoint['work_dir'] + '/' + run.name + '/references/' + key + '/' + model + '/' + stage,
                'tt_dir': config['tt']['work_dir'] + '/' + run.name + '/references/' + model + '/' + stage,
                'endpoint_sha256': digest(endpoint), 'model': model, 'stage': stage}
    record = json.loads(ready.read_text())
    if (any(record.get(k) != value for k, value in expected.items())
            or json.loads(deployed.read_text()).get('tt_dir') != expected['tt_dir']):
        return None
    return record

@ChiaFunction(resources={'reference': 1}, max_retries=0)
def reference(run_dir, stage, config, model="600m"):
    model_spec(model)
    cached = cached_reference(run_dir, stage, config, model)
    if cached:
        return cached
    run, endpoint = Path(run_dir), config['reference']
    key = digest(endpoint)[:16]
    cache = run / 'references' / key / model / stage
    directory = endpoint['work_dir'] + '/' + run.name + '/references/' + key + '/' + model + '/' + stage
    q = shlex.quote
    if not (cache / 'reference.tar').exists():
        upload(endpoint, directory, {'evaluate.py': (run / 'harness/evaluate.py').read_bytes(),
                                     'corpus.json': (run / 'corpus.json').read_bytes(),
                                     **{name: (run / 'harness' / name).read_bytes() for name in ('models.py', 'models.json')}})
        reference_job(endpoint, directory, shlex.join([endpoint['python'], 'evaluate.py', 'reference',
            '--stage', stage, '--model', model, '--corpus', 'corpus.json', '--out', '.']), True, cache / 'job.json')
        data = ssh(endpoint, 'tar -cf - -C ' + q(directory) + ' input oracle', timeout=300)
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / 'reference.tmp'
        temporary.write_bytes(data)
        temporary.replace(cache / 'reference.tar')
    target = config['tt']['work_dir'] + '/' + run.name + '/references/' + model + '/' + stage
    deployed = cache / (digest(config['tt']) + '.deployed.json')
    if not deployed.exists():
        ssh(config['tt'], 'mkdir -p ' + q(target) + ' && tar -xf - -C ' + q(target),
            (cache / 'reference.tar').read_bytes(), 300)
        write_json(deployed, {'tt_dir': target})
    result = {'reference_dir': directory, 'tt_dir': target, 'endpoint_sha256': digest(endpoint), 'model': model, 'stage': stage}
    write_json(cache / 'ready.json', result)
    return result


@ChiaFunction(resources={'tt': 1}, max_retries=0)
def experiment(run_dir, job, request, config):
    run, stage, endpoint = Path(run_dir), request.get('stage'), config['tt']
    try:
        model = request.get('model', '600m')
        public_stage = request.get('public_input_stage', '')
        if public_stage and (request.get('kind') != 'run' or public_stage not in ('smoke', 'bringup')):
            raise ValueError('public_input_stage requires run and smoke/bringup')
        input_stage = stage or public_stage
        ref = (cached_reference(run_dir, input_stage, config, model)
               or get(reference.chia_remote(run_dir, input_stage, config, model))) if input_stage else None
        transport_timeout = (full_job_timeout(endpoint) + 400
                             if request.get('kind') == 'evaluate' and stage == 'full' else 22000)
        result = remote(config, run_dir, {'job': job, **request}, timeout=transport_timeout)
        if stage and result['returncode'] == 0:
            root = endpoint['work_dir'] + '/' + run.name
            output = root + '/jobs/' + job + '/output'
            if stage != 'full':
                command = shlex.join([endpoint['python'], root + '/harness/evaluate.py', 'assess',
                    '--input', ref['tt_dir'] + '/input', '--oracle', ref['tt_dir'] + '/oracle', '--output', output])
                ssh(endpoint, command + ' || test -f ' + shlex.quote(output + '/verdict.json'), timeout=300)
                result['verdict'] = json.loads(ssh(endpoint, 'cat ' + shlex.quote(output + '/verdict.json')))
            else:
                outputs = ssh(endpoint, 'tar -cf - -C ' + shlex.quote(root + '/jobs/' + job) + ' output', timeout=300)
                gpu, score = config['reference'], ref['reference_dir'] + '/scores/' + job
                ssh(gpu, 'mkdir -p ' + shlex.quote(score) + ' && tar -xf - -C ' + shlex.quote(score), outputs, 300)
                command = shlex.join([gpu['python'], '../../evaluate.py', 'assess', '--input', '../../input',
                    '--oracle', '../../oracle', '--output', 'output']) + ' || test -f output/verdict.json'
                reference_job(gpu, score, command, False, run / 'jobs' / job / 'score-job.json')
                result['verdict'] = json.loads(ssh(gpu, 'cat ' + shlex.quote(score + '/output/verdict.json')))
        result['cluster_sha256'] = digest(config)
        result['precision'] = request.get('precision')
        result['model'] = request.get('model', '600m')
    except Exception as exc:
        result = {'returncode': -1, 'failure': 'infrastructure', 'error': str(exc)[-2000:]}
    write_json(run / 'jobs' / job / 'result.json', result)
    return result


@ChiaFunction(resources={'reference': 1}, max_retries=0)
def precision_study(run_dir, job, config, model="600m"):
    """Matched short CUDA comparisons; FP32 oracle remains immutable."""
    try:
        ref = reference(run_dir, 'bringup', config, model)
        endpoint, modes = config['reference'], {}
        for mode in ('fp32', 'fp16', 'bf16'):
            directory = ref['reference_dir'] + '/precision/' + mode
            command = shlex.join([endpoint['python'], '../../evaluate.py', 'reference-benchmark',
                '--input', '../../input', '--oracle', '../../oracle', '--output', 'output', '--precision', mode, '--model', model])
            reference_job(endpoint, directory, command + ' || test -f output/verdict.json', True,
                          Path(run_dir) / 'jobs' / job / (mode + '-job.json'))
            modes[mode] = {name: json.loads(ssh(endpoint, 'cat ' + shlex.quote(directory + '/output/' + name + '.json')))
                           for name in ('measurements', 'verdict')}
            for case in modes[mode]['measurements']['cases']:
                case.pop('translations', None)
        result = {'returncode': 0, 'modes': modes, 'cluster_sha256': digest(config),
                  'model': model, 'scope': 'eight short cases; not full-corpus quality or TT performance'}
    except Exception as exc:
        result = {'returncode': -1, 'failure': 'infrastructure', 'error': str(exc)[-2000:]}
    write_json(Path(run_dir) / 'jobs' / job / 'result.json', result)
    return result


class PortTools(ChiaTool):
    def setup(self, workspace, run_dir, model="600m"):
        model_spec(model)
        self.model = model
        self.workspace, self.run_dir = Path(workspace), Path(run_dir)
        self.refs = {}
        for name in ('files', 'read', 'write', 'run', 'evaluate', 'status', 'log', 'submit'):
            self.mcp.add_tool(getattr(self, name), name='nllb_' + name)

    def files(self) -> list:
        return list(snapshot(self.workspace))

    def read(self, path: str) -> str:
        return safe_file(self.workspace, path).read_text()[:24000]

    def write(self, path: str, content: str) -> dict:
        if Path(path).as_posix() in ('CONTRACT.md', 'TT_GUIDE.md', 'TASK.md') or (self.run_dir / 'submission.json').exists():
            raise ValueError('instructions or submitted source are read-only')
        if len(content.encode()) > 2_000_000:
            raise ValueError('file too large')
        target = safe_file(self.workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {'saved': path}

    def start(self, request):
        files = snapshot(self.workspace)
        stage = request.get('stage')
        config = activated(self.run_dir)
        request = {**request, 'model': request.get('model', getattr(self, 'model', '600m')),
                   'cluster_sha256': digest(config)}
        input_stage = stage or request.get('public_input_stage')
        if input_stage:
            request['suite_definition_sha256'] = suite_identity(self.run_dir, request.get('model', '600m'), input_stage)
        for path in (self.run_dir / 'jobs').glob('*/request.json'):
            old = json.loads(path.read_text())
            if (stage and old.get('stage') == stage and old['code_sha256'] == code_hash(files)
                    and old.get('suite_definition_sha256') == request.get('suite_definition_sha256')
                    and old.get('model', '600m') == request.get('model', '600m')
                    and old.get('precision') == request.get('precision')
                    and old.get('cluster_sha256') == digest(config)):
                result = path.with_name('result.json')
                if not result.exists() or json.loads(result.read_text()).get('verdict', {}).get('passed'):
                    return {'job': path.parent.name, 'reused': True}
        job = uuid.uuid4().hex
        write_json(self.run_dir / 'jobs' / job / 'source.json', files)
        write_json(self.run_dir / 'jobs' / job / 'request.json', {**request, 'code_sha256': code_hash(files)})
        self.refs[job] = experiment.chia_remote(str(self.run_dir), job, {**request, 'files': files}, config)
        return {'job': job, 'status': 'running'}

    def run(self, command: str, timeout_seconds: int = 120, public_input_stage: str = '', model: str = '') -> dict:
        """Async isolated TT shell. /opt/tt-metal public source; /weights pinned weights.
        model selects a registered checkpoint; empty uses the campaign's model.
        Optional public_input_stage=smoke/bringup mounts prepared public workload files
        for that model read-only at /input; no oracle or evaluation pass is exposed.
        Only write() persists source edits. No network, old ports, or credentials.
        A timeout quarantines its card, so use short focused commands, never sleep.
        """
        if public_input_stage not in ('', 'smoke', 'bringup'):
            raise ValueError('public_input_stage must be empty, smoke or bringup')
        model = model or getattr(self, 'model', '600m')
        model_spec(model)
        return self.start({'kind': 'run', 'command': command, 'timeout': min(timeout_seconds, 300),
                           'public_input_stage': public_input_stage, 'model': model})

    def evaluate(self, stage: str = 'smoke', precision: str = 'bf16', model: str = '') -> dict:
        """Independent CUDA-reference check: smoke (tiny) or bringup (short behavior suite).
        Full corpus is reserved for frozen submit(). Call status to collect results.
        """
        model = model or getattr(self, 'model', '600m')
        model_spec(model)
        if precision not in PRECISIONS:
            raise ValueError('unsupported precision')
        if stage not in ('smoke', 'bringup'):
            raise ValueError('use smoke/bringup; full is controller-only after submission')
        if not (self.workspace / 'backend.py').exists():
            raise ValueError('backend.py required')
        return self.start({'kind': 'evaluate', 'stage': stage, 'precision': precision, 'model': model})

    def status(self, compact: bool = False, job_id: str = '') -> dict:
        """Collect recent results and every pending job, including older background studies.
        compact=True bounds stdout/stderr to 1000 characters each and adds request
        provenance and retrieval hints. Structured results/gates remain complete.
        job_id retrieves one own historical request/result regardless of age, with
        complete request identity and compact log tails. This exact lookup is local
        and read-only, including pending jobs; it never reruns or refreshes a job.
        """
        if type(compact) is not bool:
            raise ValueError('compact must be a boolean')
        if not isinstance(job_id, str):
            raise ValueError('job_id must be a string')
        results = {}
        if job_id:
            if (len(job_id) > 80 or job_id[0] not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in job_id)):
                raise ValueError('invalid job ID')
            job = self.run_dir / 'jobs' / job_id
            path = job / 'request.json'
            result_file = job / 'result.json'
            if (any(p.is_symlink() for p in (job.parent, job, path, result_file))
                    or not path.is_file() or (result_file.exists() and not result_file.is_file())):
                raise ValueError('unknown or unsafe job')
            visible = [path]
        else:
            paths = sorted((self.run_dir / 'jobs').glob('*/request.json'), key=lambda p: p.stat().st_mtime)
            visible = [p for p in paths[:-12] if not p.with_name('result.json').exists()] + paths[-12:]
        for path in visible:
            result_file = path.with_name('result.json')
            request = json.loads(path.read_text())
            if result_file.exists():
                result = json.loads(result_file.read_text())
                if not job_id and request.get('stage') == 'bringup' and result.get('verdict', {}).get('passed'):
                    baseline = (self.run_dir / 'baselines' / request.get('model', '600m')
                                / request['precision'] / request['cluster_sha256'] / (result['verdict']['suite_sha256'] + '.json'))
                    if not baseline.exists():
                        write_json(baseline, {'job': path.parent.name, 'code_sha256': request['code_sha256'],
                                             'precision': request.get('precision'), 'cluster_sha256': request.get('cluster_sha256'),
                                             'model': request.get('model', '600m'), 'suite_sha256': result['verdict']['suite_sha256'],
                                             'suite_definition_sha256': request['suite_definition_sha256']})
                results[path.parent.name] = result
            elif job_id:
                results[path.parent.name] = {'status': 'pending', 'progress': {'phase': 'saved_request_without_result'}}
            else:
                try:
                    config = json.loads((self.run_dir / 'clusters' / (request['cluster_sha256'] + '.json')).read_text())
                    progress = (reference_progress(config, path.parent) if request.get('kind') == 'reference-study' else
                                remote(config, str(self.run_dir), {'kind': 'status', 'job': path.parent.name}, timeout=10))
                except Exception:
                    progress = {'phase': 'reference_or_queue'}
                results[path.parent.name] = {'status': 'pending', 'progress': progress}
            if compact or job_id:
                item = dict(results[path.parent.name])
                item['request_identity'] = {key: request.get(key) for key in
                    ('kind', 'stage', 'model', 'precision', 'code_sha256', 'cluster_sha256',
                     'suite_definition_sha256')}
                if job_id:
                    item['request_identity'] = request
                tails = {}
                for stream in ('stdout', 'stderr'):
                    if isinstance(item.get(stream), str):
                        text = item[stream]
                        item[stream] = text[-1000:]
                        tails[stream] = {'available_chars': len(text), 'returned_chars': len(item[stream]),
                                         'truncated': len(text) > 1000}
                item['log_tails'] = tails
                item['details'] = ({'tool': 'nllb_log', 'job_id': path.parent.name,
                                    'streams': ['stdout', 'stderr'],
                                    'note': 'Saved result excerpts may already be truncated; log reads original byte pages.'}
                                   if request.get('kind') in ('run', 'evaluate') else
                                   {'tool': 'nllb_status', 'compact': False, 'job_id': path.parent.name,
                                    'note': 'Reference-study details use status; TT log access is unavailable.'})
                results[path.parent.name] = item
        return results

    def log(self, job_id: str, stream: str = 'stdout', offset_bytes: int = 0, limit_bytes: int = 65536) -> dict:
        """Read existing TT job stdout/stderr without rerunning it. No oracle access.
        Byte pages use UTF-8 replacement decoding (a split character can be replaced).
        Use next_offset_bytes to continue; eof and size_bytes describe this read's snapshot.
        """
        validate_log_request(job_id, stream, offset_bytes, limit_bytes)
        job = self.run_dir / 'jobs' / job_id
        path = job / 'request.json'
        if any(p.is_symlink() for p in (job.parent, job, path)) or not path.is_file():
            raise ValueError('unknown or unsafe job')
        request = json.loads(path.read_text())
        if request.get('kind') not in ('run', 'evaluate'):
            raise ValueError('only own TT run/evaluate job logs are exposed')
        key = request.get('cluster_sha256', '')
        if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('missing archived cluster identity')
        archive = self.run_dir / 'clusters' / (key + '.json')
        if archive.parent.is_symlink() or archive.is_symlink() or not archive.is_file():
            raise ValueError('missing or unsafe archived cluster')
        config = json.loads(archive.read_text())
        if digest(config) != key:
            raise ValueError('archived cluster identity mismatch')
        return remote(config, str(self.run_dir), {'kind': 'log', 'job': job_id, 'stream': stream,
                      'offset_bytes': offset_bytes, 'limit_bytes': limit_bytes}, timeout=15)

    def submit(self, precision: str = 'bf16', model: str = '') -> dict:
        """Freeze source after bringup passes. Controller alone runs the full acceptance corpus."""
        model = model or getattr(self, 'model', '600m')
        model_spec(model)
        if precision not in PRECISIONS:
            raise ValueError('unsupported precision')
        files = snapshot(self.workspace)
        config = activated(self.run_dir)
        source, signature = code_hash(files), digest(config)
        definition = suite_identity(self.run_dir, model, 'bringup')
        requests = sorted((self.run_dir / 'jobs').glob('*/request.json'),
                          key=lambda path: path.stat().st_mtime, reverse=True)
        for path in requests:
            req = json.loads(path.read_text())
            result = path.with_name('result.json')
            if (req.get('kind') == 'evaluate' and req.get('stage') == 'bringup'
                    and req.get('code_sha256') == source
                    and req.get('suite_definition_sha256') == definition
                    and req.get('model', '600m') == model and req.get('precision') == precision
                    and req.get('cluster_sha256') == signature):
                if result.exists() and passing_evaluation(json.loads(result.read_text()), path.parent.name,
                                                         signature, model, precision, 'bringup'):
                    write_json(self.run_dir / 'submission.json', {'files': files, 'code_sha256': source, 'precision': precision,
                        'cluster_sha256': signature, 'model': model, 'suite_definition_sha256': definition})
                    return {'status': 'frozen_for_full_evaluation'}
                # A later matching failure, pending job or incomplete receipt must
                # not be bypassed by an older pass for the same source/configuration.
                break
        raise ValueError('current source needs an independent bringup pass before submission')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cluster', type=Path, default=HERE / 'cluster.yaml')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=6)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--model', default='600m')
    parser.add_argument('--seed-run', type=Path)
    parser.add_argument('--seed-job', help='Use this passing bringup snapshot rather than a changing workspace')
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    if args.seed_job and not args.seed_run:
        parser.error('--seed-job requires --seed-run')
    model_spec(args.model)
    if not 0 < args.hours <= 24:
        parser.error('hours must be in (0,24]')
    run = (args.run_dir or HERE / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if args.detach:
        if run.exists():
            parser.error('run directory already exists')
        run.parent.mkdir(parents=True, exist_ok=True)
        with run.with_suffix('.log').open('wb') as log:
            command = [sys.executable, __file__, '--corpus', str(args.corpus.resolve()),
                '--hours', str(args.hours), '--run-dir', str(run), '--cluster', str(args.cluster.resolve()), '--model', args.model]
            if args.seed_run:
                command += ['--seed-run', str(args.seed_run.resolve())]
            if args.seed_job:
                command += ['--seed-job', args.seed_job]
            if args.reference_run:
                command += ['--reference-run', str(args.reference_run.resolve())]
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
        if sys.platform == 'darwin':
            subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(child.pid)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        write_json(run.with_suffix('.launch.json'), {'pid': child.pid, 'run': str(run)})
        print(json.dumps({'pid': child.pid, 'run': str(run)}))
        return
    if not args.controller:
        load_cluster(args.cluster)  # Validate before creating a campaign archive.
        corpus = json.loads(args.corpus.read_text())
        if digest(corpus) != CORPUS_HASH:
            raise ValueError('corpus identity mismatch')
        repository = repository_root()
        if repository == Path('/'):
            raise ValueError('explicit repository isolation root required')
        run.mkdir(parents=True)
        workspace = Path(tempfile.mkdtemp(prefix='nllb-fresh-'))
        for name in ('CONTRACT.md', 'TT_GUIDE.md', 'TASK.md'):
            (workspace / name).write_bytes((HERE / name).read_bytes())
        task = workspace / 'TASK.md'
        task.write_text('# Selected campaign model\n\nThis campaign targets ' + model_spec(args.model)['repo_id']
                        + ' (model key ' + args.model + '). This selection overrides size-specific defaults below.\n\n'
                        + task.read_text())
        provenance = seed_workspace(workspace, args.seed_run, args.model, args.seed_job) if args.seed_run else {}
        (run / 'harness').mkdir()
        for path in HERE.iterdir():
            if path.is_file() and path.suffix in ('.py', '.md', '.json'):
                (run / 'harness' / path.name).write_bytes(path.read_bytes())
        write_json(run / 'corpus.json', corpus)
        manifest = {'workspace': str(workspace), 'pid': os.getpid(), 'repository_root': str(repository),
                    'corpus_sha256': digest(corpus), 'harness_sha256': harness_identity(run / 'harness'),
                    'seed_files': sorted(p.name for p in workspace.iterdir()), 'budget_hours': args.hours,
                    'cluster_file': str(args.cluster.resolve()), 'model': args.model,
                    'seed_policy': 'fresh-public-nllb',
                    'reference_run': str(args.reference_run.resolve()) if args.reference_run else None, **provenance}
        write_json(run / 'manifest.json', manifest)
        command = [sys.executable, str(run / 'harness/loop.py'), '--controller', '--run-dir', str(run),
                   '--cluster', str(args.cluster.resolve()), '--corpus', str(run / 'corpus.json'),
                   '--hours', str(args.hours), '--model', args.model]
        if args.reference_run:
            command += ['--reference-run', str(args.reference_run.resolve())]
        os.execve(sys.executable, command, frozen_environment(run, manifest))
    manifest = json.loads((run / 'manifest.json').read_text())
    if (run / 'status.json').exists():
        raise ValueError('existing controller state requires explicit recovery; this is not resume')
    if args.model != manifest['model']:
        raise ValueError('controller model differs from archived selection')
    workspace = Path(manifest['workspace'])
    if not workspace.is_absolute() or not workspace.is_dir():
        raise ValueError('saved fresh workspace missing')
    corpus = json.loads((run / 'corpus.json').read_text())
    if digest(corpus) != manifest['corpus_sha256']:
        raise ValueError('archived corpus identity mismatch')
    config = load_cluster(args.cluster)
    if args.model not in config['tt'].get('weights_by_model', {'600m': config['tt'].get('weights')}):
        raise ValueError('selected model has no configured TT checkpoint')
    def state(value, **extra):
        write_json(run / 'status.json', {'status': value, 'time': datetime.now(timezone.utc).isoformat(), **extra})
    state('preflight')
    ray.init(num_cpus=8, include_dashboard=False, _node_ip_address='127.0.0.1',
             runtime_env={'env_vars': {'PYTHONPATH': str(run / 'harness'),
                                      'NLLB_REPOSITORY_ROOT': manifest['repository_root']}},
             resources={'codex_creds': 1, 'tt': int(config['tt']['resources'].get('tt', 1)), 'reference': 1})
    from chia.trace.profiler import start_collector
    start_collector(log_dir=str(run / 'chia-profile'))
    tools = None
    try:
        health = prepare(run, config)
        if manifest.get('reference_run'):
            import_references(run, manifest['reference_run'], config, args.model)
        tools = PortTools(name=tool_name(run), workspace=str(workspace), run_dir=str(run), model=args.model)
        preflight = get(worker_node.chia_remote(str(workspace), str(run), [tools], 0,
            'Infrastructure check only: list files, read CONTRACT.md, confirm nllb_read ../outside is rejected. '
            'Do not implement or evaluate yet. End with PREFLIGHT_OK only if scoped tools work.'))
        if not preflight['success'] or not preflight['response'].strip().endswith('PREFLIGHT_OK'):
            state('paused_agent_preflight', error=preflight['error'])
            return
        state('reference_preflight')
        try:
            get(reference.chia_remote(str(run), 'smoke', config, args.model))
        except Exception as exc:
            state('paused_reference', error=str(exc)[-1500:])
            return
        study_job = 'gpu-precision-' + args.model + '-' + digest(config)[:12]
        write_json(run / 'jobs' / study_job / 'request.json',
                   {'kind': 'reference-study', 'stage': 'precision-study', 'model': args.model, 'cluster_sha256': digest(config)})
        study = precision_study.chia_remote(str(run), study_job, config, args.model)
        end, turn, retries, timeouts = time.monotonic() + args.hours * 3600, 0, 0, 0
        previous = None
        while time.monotonic() < end:
            turn += 1
            if (run / 'submission.json').exists():
                break
            config, change = refresh_cluster(run, args.cluster, config)
            if change == 'draining':
                state('draining_for_cluster_change')
                time.sleep(10)
                continue
            if change == 'changed':
                state('checking_replacement_reference')
                get(reference.chia_remote(str(run), 'smoke', config, args.model))
            health = remote(config, str(run), {'kind': 'health'})
            if not any(c['state'] == 'healthy' for c in health['cards']):
                state('checking_timeout_recovery', health=health)
                health = recover_timed_out_card(run, config, health, args.model)
            if not any(c['state'] == 'healthy' for c in health['cards']):
                state('paused_hardware', health=health)
                break
            facts = supervisor_facts(workspace, run, config, args.model)
            state('supervising', turn=turn, health=health, recent_jobs=facts['recent_jobs'])
            review = get(supervisor_node.chia_remote(str(workspace), str(run), turn,
                {**facts, 'health':health, 'worker_report':(previous or {}).get('response','')[-2000:],
                 'worker_error':(previous or {}).get('error',''),
                 'remaining_minutes':int((end-time.monotonic())/60)}))
            if not review['success']:
                retries += 1
                if review['error'] != 'capacity' or retries > 6:
                    state('paused_supervisor', error=review['error']); break
                state('retry_wait', reason='capacity', retry=retries)
                time.sleep(min(30*retries, 60)); continue
            if review['decision']['action'] == 'pause':
                state('paused_supervisor', instruction=review['decision']['instruction']); break
            state('working', turn=turn, instruction=review['decision']['instruction'], health=health)
            previous = get(worker_node.chia_remote(str(workspace), str(run), [tools], turn,
                review['decision']['instruction'] + '\nContinue from your saved STATE.md. Call nllb_status(compact=True) before repeating tests.'))
            if not previous['success']:
                if previous['error'] == 'timeout':
                    timeouts += 1
                    if timeouts <= 3:
                        state('worker_timeout_recovery', turn=turn, retry=timeouts)
                        continue
                retries += 1
                if previous['error'] != 'capacity' or retries > 6:
                    state('paused_worker', error=previous['error']); break
                time.sleep(min(30*retries,60)); continue
            retries = timeouts = 0
        else:
            state('budget_exhausted')
        if (run / 'submission.json').exists():
            state('full_validation', turn=turn)
            submission = json.loads((run / 'submission.json').read_text())
            frozen = json.loads((run / 'clusters' / (submission['cluster_sha256'] + '.json')).read_text())
            result = get(experiment.chia_remote(str(run), 'final', {'kind':'evaluate','stage':'full',
                'files':submission['files'], 'precision':submission['precision'], 'model':submission.get('model', '600m')}, frozen))
            state('accepted_research_port' if result.get('verdict',{}).get('passed') else 'full_failed')
    except BaseException as exc:
        state('controller_error', error=str(exc)[-1500:])
        raise
    finally:
        if tools:
            tools.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()
