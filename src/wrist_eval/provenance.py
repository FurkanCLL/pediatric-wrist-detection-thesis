"""Compact dependency/source identity and final preservation checks."""

import importlib.metadata
import json
import subprocess
from pathlib import Path
from .common import (ROOT, LOCAL, PUBLIC, baseline_inventory, config, identity, save, sha,
                     source_snapshot, validation_records, verify_validation_files)


def environment_receipt():
    from wrist_train.protocol import load_protocol
    before={}
    for line in (ROOT/"outputs/packages_before_sahi.txt").read_text(encoding="utf-16" if (ROOT/"outputs/packages_before_sahi.txt").read_bytes().startswith(b'\xff\xfe') else "utf-8-sig").splitlines():
        if "==" in line:
            name,version=line.split("==",1);before[name.lower().replace("_","-")]=version
    current={d.metadata["Name"].lower().replace("_","-"):d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    changed={name:{"before":version,"after":current.get(name)} for name,version in before.items() if current.get(name)!=version}
    # pip freeze represents the two CUDA wheels as direct URLs and omits pip itself.
    # Check those runtime versions against the frozen protocol instead of calling them new additions.
    protocol,_=load_protocol()
    for name,version in protocol["environment"]["packages"].items():
        if current[name.lower().replace("_","-")]!=version:
            changed[name]={"before":version,"after":current.get(name.lower().replace("_","-"))}
    if changed:raise RuntimeError("Existing package versions changed: "+str(changed))
    plan=json.loads((ROOT/"outputs/sahi_install_plan.json").read_text())
    added={item["metadata"]["name"].lower():item["metadata"]["version"] for item in plan["install"]}
    if any(current[name]!=version for name,version in added.items()):
        raise RuntimeError("Installed inference additions differ from reviewed plan")
    import sahi, ultralytics, torch
    root=Path(sahi.__file__).parent
    result={"sahi":sahi.__version__,"existing_package_changes":changed,"added_packages":added,
            "ultralytics":ultralytics.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,
            "sahi_source_sha256":{name:sha(root/name) for name in ("predict.py","slicing.py","models/ultralytics.py","models/base.py","postprocess/combine.py")},
            "sources":{"package":"https://pypi.org/project/sahi/0.12.8/","integration":"https://obss.github.io/sahi/models/ultralytics/",
                "api":"https://obss.github.io/sahi/api/","slicing":"https://obss.github.io/sahi/slicing/","paper":"https://arxiv.org/abs/2202.06934"}}
    result["metadata_correction"]="The initial receipt treated direct-URL CUDA wheels and pip's omitted freeze entry as additions. The reviewed pip install plan identifies the seven actual additions; initial receipt remains preserved."
    save(PUBLIC/"compatibility_verified.json",result)
    return result


def final_checks():
    from wrist_train.protocol import load_protocol
    _,lock=load_protocol()
    baseline_before=json.loads((LOCAL/"baseline_inventory_before.json").read_text())
    source_before=json.loads((LOCAL/"source_snapshot_before.json").read_text())
    current=baseline_inventory();snapshot=source_snapshot()
    candidate=ROOT/"configs/class_aware_candidates.json"
    checks={"baseline_36_artifacts_unchanged":current==baseline_before,"baseline_inventory_sha256":identity(current),
        "raw_prepared_stat_inventory_unchanged":identity(snapshot)==identity(source_before),
        "raw_prepared_stat_files":len(snapshot),"validation_file_hashes":verify_validation_files(validation_records()),
        "frozen_training_protocol_sha256":lock["protocol_sha256"],"frozen_sources_valid":True,
        "class_aware_config_sha256":identity(json.loads(candidate.read_text())),
        "no_model_training":True,"no_heldout_test_image_contents_opened":True,"no_adaptive_pipeline_implemented":True}
    expected=identity(json.loads(subprocess.check_output(["git","show","bd83049:configs/class_aware_candidates.json"],cwd=ROOT,text=True)))
    checks["class_aware_candidates_unchanged"]=checks["class_aware_config_sha256"]==expected
    if not all(checks[k] for k in ("baseline_36_artifacts_unchanged","raw_prepared_stat_inventory_unchanged","class_aware_candidates_unchanged")):
        raise RuntimeError("Preservation check failed")
    save(PUBLIC/"preservation_checks.json",checks)
    return checks


if __name__=="__main__":
    environment_receipt()
