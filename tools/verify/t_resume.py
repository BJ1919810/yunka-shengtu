"""离线验证续跑：job.sh 生成 + sanitize_opts 校验。不连云卡。"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "train_web"))
os.chdir(ROOT)
import server

os.makedirs("_selftest_sh", exist_ok=True)

def show(tag, o, job="rs0001"):
    ok, msg = server.sanitize_opts(dict(o))
    if not ok:
        print("[%s] 被拒: %s" % (tag, msg))
        return None
    t = server.build_job_sh(ok, job)
    open("_selftest_sh/%s.sh" % tag, "w", encoding="utf-8", newline="\n").write(t)
    line = [l for l in t.splitlines() if "resume_from_checkpoint" in l]
    print("[%s] 通过 | %s" % (tag, line[0].strip() if line else "(未加 resume 参数)"))
    return t

B = dict(server.DEFAULTS)

# 1. 不续跑（默认）
show("a_none", B)

# 2. latest
o = dict(B); o["resume"] = "latest"; o["steps"] = 1000
show("b_latest", o)

# 3. 指定 checkpoint
o = dict(B); o["resume"] = "checkpoint-500"; o["steps"] = 800
show("c_ckpt", o)

# 4. 坏值：绝对路径（应被拒）
o = dict(B); o["resume"] = "/root/autodl-tmp/lora-out/bj_style/checkpoint-500"
show("d_abspath", o)

# 5. 坏值：目录名不含 checkpoint-
o = dict(B); o["resume"] = "checkpoint-abc"
show("e_badname", o)

# 6. 演练 + 续跑（应被拒）
o = dict(B); o["resume"] = "latest"; o["dry"] = True
show("f_dry_resume", o)

print()
print("=== bash -n 语法检查 ===")
import subprocess
for f in sorted(os.listdir("_selftest_sh")):
    p = os.path.join("_selftest_sh", f)
    r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL "), f, r.stderr.strip()[:80])
