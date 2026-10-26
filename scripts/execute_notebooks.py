from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent / "ml_20206 (1).ipynb"
EXPECTED_ORIGINAL="ce01a450d6a9a56eb337ccf3906e6e5eafb1501951e907308cb8767fc6db2c97"


def execute():
    original_hash=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    report={"launcher_interpreter":sys.executable,"kernel":"seismic-python312","original_sha256_before":original_hash,
            "initial_reference_sha256":EXPECTED_ORIGINAL,"matches_initial_reference":original_hash==EXPECTED_ORIGINAL,"notebooks":[]}
    target=ROOT/"reports/validation/notebook_execution.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1 and target.exists():
        report=json.loads(target.read_text())
        report.pop("status",None)
    paths=sorted((ROOT/"notebooks").rglob("*.ipynb"))
    for path in paths:
        if len(sys.argv)>1 and int(path.stem.split("_")[0]) not in [int(value) for value in sys.argv[1:]]:
            continue
        print(f"RUN {path.relative_to(ROOT)}",flush=True)
        nb=nbformat.read(path,as_version=4)
        nbformat.validate(nb)
        started=time.monotonic()
        try:
            NotebookClient(nb,timeout=1800,kernel_name="seismic-python312",resources={"metadata":{"path":str(path.parent)}}).execute()
            nbformat.write(nb,path)
            errors=[out for cell in nb.cells if cell.cell_type=="code" for out in cell.get("outputs",[]) if out.output_type=="error"]
            if errors:
                raise RuntimeError(f"Output error tersimpan: {path.name}")
            result={"notebook":str(path.relative_to(ROOT)),"status":"PASS","seconds":round(time.monotonic()-started,2),
                    "code_cells":sum(c.cell_type=="code" for c in nb.cells),"executed_cells":sum(c.cell_type=="code" and c.execution_count is not None for c in nb.cells)}
        except Exception as exc:
            nbformat.write(nb,path)
            result={"notebook":str(path.relative_to(ROOT)),"status":"FAIL","error":str(exc)}
            report["notebooks"].append(result)
            target.write_text(json.dumps(report,indent=2),encoding="utf-8")
            raise
        report["notebooks"]=[r for r in report["notebooks"] if r["notebook"]!=result["notebook"]]+[result]
        report["notebooks"].sort(key=lambda r:r["notebook"])
        target.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"PASS {path.name}: {result['seconds']} s",flush=True)
    report["original_sha256_after"]=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    report["original_unchanged"]=report["original_sha256_after"]==original_hash
    report["status"]="PASS"
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Semua notebook lulus; hash notebook asli sama selama eksekusi: {report['original_unchanged']}.",flush=True)


if __name__=="__main__":
    execute()
