"""端到端验证续跑：调 /api/job/start 走续跑分支。
⚠️ 需先起服务：train_web/server.py 7870
⚠️ 无卡模式：C 用例会真启动（然后因无 GPU 失败），脚本会自动停止。
★ 跑完记得清理云端残留（lora-out/no_such_dir_xyz、lora-train/runs/*）。
"""
import json, urllib.request as U, time

B = "http://127.0.0.1:7870"

def post(p, data):
    req = U.Request(B + p, data=json.dumps(data).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with U.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode("utf-8"))
    except U.HTTPError as e:
        return {"_http": e.code, "body": e.read().decode("utf-8")[:400]}

def case(tag, body):
    d = post("/api/job/start", body)
    if d.get("_http"):
        print("[%s] HTTP %s -> %s" % (tag, d["_http"], d["body"][:200]))
    elif d.get("ok"):
        j = d["job"]
        print("[%s] ✅ 被接受（说明全部校验通过）" % tag)
        print("       job=%s out=%s pid=%s" % (j["job_id"], j["out_dir"], j["pid"]))
        return j
    else:
        print("[%s] 被拒: %s" % (tag, d.get("error")))
    return None

base = {"name": "bj_style", "trigger": "bj_style", "steps": 1000}

case("A_目录不存在", dict(base, resume="latest", name="no_such_dir_xyz"))
case("B_步数太小", dict(base, resume="latest", steps=500))

j = case("C_合理续跑", dict(base, resume="checkpoint-500", steps=800))
if j:
    print("       等 6 秒看它是否真的在跑（无卡模式预期会失败）…")
    time.sleep(6)
    st = json.loads(U.urlopen(B + "/api/job?offset=0", timeout=300).read())
    log = st.get("log") or ""
    tail = [l for l in log.splitlines() if l.strip()][-6:]
    print("       alive=%s rc=%s" % (st.get("alive"), st.get("rc")))
    for l in tail:
        print("       |", l[:150])
    print("       停止:", post("/api/job/stop", {}).get("ok"))
