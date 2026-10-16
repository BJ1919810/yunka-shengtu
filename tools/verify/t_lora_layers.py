"""离线验证 LoRA 作用面（--lora_layers）的校验与 job.sh 生成。不连云卡。

用法：  cd tools/verify && python t_lora_layers.py
背景：  .skills/79-LoRA作用面与rank.md

★ 为什么这两个必须写在白名单里：
  PEFT 0.21.2 (tuners_utils.py:2377)
      target_module_found = any(key.endswith("." + t) for t in target_modules)
  是 endswith ⇒ 写短名 "proj" 会同时命中 img_mlp.proj 和 txt_mlp.proj
  （实测验证过）。写错不报错、只是悄悄训了别的层。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "train_web"))
import server  # noqa: E402

ATTN = "to_k,to_q,to_v,to_out.0"
IMG = "img_mlp.gate_layer,img_mlp.proj,img_mlp.out"

# (名字, 输入, 期望通过?, 期望归一化后的值)
CASES = [
    ("默认 4 层 attn",            {"lora_layers": ATTN},              True,  ATTN),
    ("推荐 7 层（+img_mlp）",      {"lora_layers": ATTN + "," + IMG},   True,  ATTN + "," + IMG),
    ("空 = 走脚本默认",           {"lora_layers": ""},                True,  ""),
    ("带空格自动 trim",           {"lora_layers": " to_q , to_k "},   True,  "to_q,to_k"),
    ("短路名 proj（应拒）",        {"lora_layers": "proj"},            False, None),
    ("短路名 out（应拒）",         {"lora_layers": "out"},             False, None),
    ("1.0 命名 net.0.proj（应拒）", {"lora_layers": "img_mlp.net.0.proj"}, False, None),
    ("1.0 命名 add_q_proj（应拒）", {"lora_layers": "add_q_proj"},      False, None),
    ("重复项（应拒）",             {"lora_layers": "to_q,to_q"},        False, None),
    ("瞎写（应拒）",               {"lora_layers": "hello,world"},     False, None),
    ("txt_mlp（2.1 无，但白名单放行）", {"lora_layers": "txt_mlp.proj"},  True,  "txt_mlp.proj"),
]


def main() -> int:
    fails = 0
    print("=" * 66)
    print("一、sanitize_opts 校验")
    print("=" * 66)
    for i, (tag, inp, want_ok, want_val) in enumerate(CASES, 1):
        o, err = server.sanitize_opts({"name": "t_%d" % i, **inp})
        got_ok = o is not None
        mark = "✓" if got_ok == want_ok else "✗"
        if got_ok != want_ok:
            fails += 1
        detail = repr(o["lora_layers"]) if o else ("拒: " + err.split("\n")[0])
        print("  %s %-26s %s" % (mark, tag, detail))
        if o and want_val is not None and o["lora_layers"] != want_val:
            print("      ✗ 归一化不符，期望 %r" % want_val)
            fails += 1

    print()
    print("=" * 66)
    print("二、job.sh 参数生成")
    print("=" * 66)
    # 传了 lora_layers ⇒ 必须出现在命令行里
    o, _ = server.sanitize_opts({"name": "t_sh", "lora_layers": ATTN + "," + IMG})
    sh = server.build_job_sh(o, "T")
    has = "--lora_layers" in sh
    want_line = "--lora_layers '%s'" % (ATTN + "," + IMG)
    has_line = want_line in sh
    print("  %s 含 --lora_layers 开关" % ("✓" if has else "✗"))
    print("  %s 参数值精确匹配（带引号）" % ("✓" if has_line else "✗"))
    fails += 0 if (has and has_line) else 1
    for ln in sh.split("\n"):
        if "--lora_layers" in ln or "--rank " in ln or "--lora_alpha " in ln:
            print("       |" + ln)

    # 空 ⇒ 不传（走脚本默认）
    o2, _ = server.sanitize_opts({"name": "t_sh2", "lora_layers": ""})
    sh2 = server.build_job_sh(o2, "T")
    no_flag = "--lora_layers" not in sh2
    print("  %s 空值时**不传**该参数（走脚本默认 4 层）" % ("✓" if no_flag else "✗"))
    fails += 0 if no_flag else 1

    # bash -n 语法
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                     encoding="utf-8", newline="\n") as f:
        f.write(sh)
        tmp = f.name
    rc = subprocess.call(["bash", "-n", tmp])
    os.unlink(tmp)
    print("  %s bash -n 语法检查（rc=%d）" % ("✓" if rc == 0 else "✗", rc))
    fails += 0 if rc == 0 else 1

    print()
    print("=" * 66)
    print("全部通过 ✅" if fails == 0 else "有 %d 处不符 ❌" % fails)
    print("=" * 66)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
