# Stage-216来源包换行兼容修复

2026-11-28。用户在 `bundle(args.stage215_bundle)` 处收到 `Need reviewed Stage-215 bundle`，发生在来源验证、manifest发布和GPU推理之前。

已在本机重现原因：215原始TXT混用LF与CRLF，628082字节、2783处CRLF，SHA256为 `275f66a26c14458045ca5af41a1831ac2537242fb77223df19a0cf6566067f87`。归档漏配 `.gitattributes -text`，实际 `git show HEAD:<结果包路径>` 得到全LF版本，其SHA256为 `a7bad86f45740e8b93742eabf34b4154dcc65567c5b3377fe4f0831c7d73ec55`。与原始包仅换行差异一致，不是新的实验结果。

修复216入口：仅将CRLF换成LF后，对整个文件校验固定的已复核规范化哈希；不删除其他空白、不忽略多余字节、不接受BOM或数值改动。之后仍核JSON指纹、来源清单和各CSV原声明哈希；表段只尝试LF/CRLF换行重建及既有分隔符边界，不跳过表校验。原始包、全LF、全CRLF均可还原同一decision/plan，数字变更、额外换行、额外字节和BOM被拒绝。

同时为215结果归档补 `-text`，为两个新增入口及测试/启动脚本补LF规则。已经存在的Git对象不会被属性修改自动改写，因此216仍需兼容已同步的全LF包。不代为改Git索引、提交或推送，不修改Stage1–215代码/模型/冻结协议。

5项完整CPU回归通过（46.337秒），含上述换行/篡改案例、真实CENTER配合合成骨干、缓存与独立spawn、两源完整收集/汇总/封存。Python3.10语法通过；没有新GPU或精度结果。机器记录见[校验JSON](results/stage216_bundle_encoding_fix_validation_20261128.json)。

## 无需等待代码同步的服务器处理

原服务器生成包默认位于：

```text
output/stage215/same_model_gpu_replay_v1/stage215_results_20261128T102259Z_26003.txt
```

显式用 `--stage215-bundle` 指向该原包，可让旧入口继续执行；启动前先核其原始SHA256必须为上面的275f66a2…值。启动脚本已将额外CLI参数原样透传。换新日志，不清模型/缓存，不解除真实写入锁；本次报错在新manifest发布前，一般可以使用原216输出目录重试。

代码补丁按Git流程同步后，默认归档的LF版本也可通过同一固定内容校验。若原包或规范化哈希仍不符，应停止并核对文件路径/来源，不扩大容差。
