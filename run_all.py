from pathlib import Path
import subprocess
import sys
import os
import time

ROOT = Path(__file__).resolve().parent

env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run([sys.executable, str(ROOT / "src" / "baselines" / "assemble.py")], cwd=ROOT, env=env, check=True)
subprocess.run([sys.executable, "-m", "src.table_pipeline", "derived"], cwd=ROOT, env=env, check=True)
scripts = sorted((ROOT / "scripts").glob("fig*.py"))
if len(scripts) != 21:
    raise RuntimeError(f"Expected 21 computational figure scripts, found {len(scripts)}")
run_started = time.time()
expected = {script.stem + ".pdf" for script in scripts}
pdf_dir = ROOT / "figures" / "pdf"
before = {name: ((pdf_dir / name).stat().st_mtime_ns if (pdf_dir / name).exists() else None) for name in expected}
for script in scripts:
    subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=True)
    target = pdf_dir / f"{script.stem}.pdf"
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"{script.name} did not produce its declared output {target.name}")
    if before[target.name] is not None and target.stat().st_mtime_ns <= before[target.name]:
        raise RuntimeError(f"{script.name} did not rewrite its declared output {target.name}")
drawn_figure = ROOT / "figures" / "pdf" / "fig01_medi_pipeline.pdf"
if not drawn_figure.exists() or drawn_figure.stat().st_size == 0:
    raise RuntimeError("Missing author-drawn Figure 1a asset")
pdfs = sorted(pdf_dir.glob("*.pdf"))
generated = {p.name for p in pdfs if p.name != "fig01_medi_pipeline.pdf" and p.stat().st_mtime >= run_started - 2}
if len(pdfs) != 22 or any(p.stat().st_size == 0 for p in pdfs) or generated != expected:
    missing = sorted(expected - generated)
    stale = sorted(generated - expected)
    raise RuntimeError(f"Expected 21 fresh computational PDFs plus Figure 1; missing={missing}, unexpected={stale}")
paper = ROOT / "paper"
for tool in ("pdflatex", "bibtex"):
    if __import__("shutil").which(tool) is None:
        raise RuntimeError(f"Required tool '{tool}' was not found on PATH")
for _ in range(1):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=paper, check=True)
subprocess.run(["bibtex", "main"], cwd=paper, check=True)
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=paper, check=True)
print(f"Generated {len(pdfs)} figure PDFs and {paper / 'main.pdf'}")
