"""
comfyui-h3-vram-autorelease
===========================
2卡模式专用:每次 prompt 结束后,杀掉 RayLight worker 释放显存,并让 RayLight 节点缓存失效,
下次点「运行」时整条 Ray 链重新执行(新 actor + 重新加载模型)。

为什么要这个:
- FSDP 分片权重常驻 GPU(每卡 ~10G),RayLight 内置的 clear_vram_after_sampling
  对 FSDP 模型无效(unpatch 只处理 patch 权重,FSDP 分片动不了)。
- 不卸的话:热着 12G/卡,下次跑片主进程再要显存 → aimdo "VRAM grow failed" / 显存不足。
- 卸 worker 后,ComfyUI 执行缓存里还留着死的 actor 句柄 → 下次跑会 RayActorError。
  所以同时把 RayLight 类节点的缓存条目清掉,下次强制重跑。

生效条件:环境变量 H3_VRAM_AUTORELEASE=1(由 scripts/start_comfyui_2gpu.sh 设置)。
四卡模式不设这个变量,插件完全休眠,四卡版行为不变。

代价:每次跑片多一次 Ray 重启 + 模型重载(约 +2~3 分钟)。换来每次跑片都从干净显存开始,
不会再出现"点运行显存不足"。
"""
import os
import shutil
import threading

ENABLED = os.environ.get("H3_VRAM_AUTORELEASE", "0") == "1"

RAY_NAMESPACE = "default"
RAY_TMPDIR = "/tmp/raylight-ray"


def _is_raylight(class_type):
    if not class_type:
        return False
    return class_type.startswith(
        ("Ray", "XFuser", "UnifiedParallel", "DPSampler", "DPKSampler")
    )


class H3VramReleaseProvider:
    """CacheProvider:追踪 RayLight 节点 + 在 prompt 结束时清缓存 + 杀 worker。"""

    def __init__(self):
        self._seen = {}          # node_id -> class_type (当前 prompt 见过的)
        self._lock = threading.Lock()

    # ---- 必须实现(全部 no-op,只用来搭钩子) ----
    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value):
        pass

    # ---- 每个节点的缓存查询/存储都会经过这里 ----
    def should_cache(self, context, value=None):
        try:
            if _is_raylight(getattr(context, "class_type", None)):
                with self._lock:
                    self._seen[context.node_id] = context.class_type
        except Exception:
            pass
        return True  # 不改变外部缓存行为(本来也没接外部缓存)

    # ---- prompt 结束(成功或报错都在 finally 里触发) ----
    def on_prompt_end(self, prompt_id):
        t = threading.Thread(target=self._release, daemon=True)
        t.start()

    def _release(self):
        try:
            self._evict_raylight_cache()
        except Exception as e:
            print(f"[h3-vram-release] 缓存失效失败(不影响出片): {e}", flush=True)
        try:
            self._kill_ray_workers()
        except Exception as e:
            print(f"[h3-vram-release] 杀 worker 失败: {e}", flush=True)

    def _evict_raylight_cache(self):
        import execution as _exec
        ex = _exec._h3_last_executor
        if ex is None or getattr(ex, "caches", None) is None:
            return
        outputs = ex.caches.outputs
        if not getattr(outputs, "initialized", False):
            return
        key_set = getattr(outputs, "cache_key_set", None)
        with self._lock:
            ray_nodes = dict(self._seen)
            self._seen.clear()
        if not ray_nodes:
            return
        removed = 0
        for nid, ct in ray_nodes.items():
            try:
                k = key_set.get_data_key(nid)
                if outputs.cache.pop(k, None) is not None:
                    removed += 1
            except Exception:
                pass
        if removed:
            print(f"[h3-vram-release] 已失效 {removed} 个 RayLight 缓存条目,下次运行将重新起 Ray+加载模型", flush=True)

    def _kill_ray_workers(self):
        import ray
        killed, how = 0, ""
        try:
            if ray.is_initialized():
                try:
                    for i in range(8):  # 最多 8 张卡,按需存在
                        try:
                            a = ray.get_actor(f"RayWorker:{i}", namespace=RAY_NAMESPACE)
                            ray.get(a.kill.remote(), timeout=120)
                            killed += 1
                        except Exception:
                            break
                except Exception as e:
                    how = f"actor kill 异常: {e}"
                try:
                    ray.shutdown()  # 本地 Ray 集群跟 driver 走, 一拆 worker 全退、显存全还
                except Exception:
                    pass
            else:
                how = "主进程 ray 上下文已不存在(集群已整体拆除)"
        except Exception as e:
            how = f"ray 清理异常: {e}"
        # 兜底: 按进程名清残留 (只认 ray::RayWorker, 不碰其他常驻服务的进程)
        leftover = self._force_kill_leftover()
        if killed:
            print(f"[h3-vram-release] 已释放 {killed} 个 Ray worker,显存已归还(下次运行重新加载)", flush=True)
        print(f"[h3-vram-release] 收尾: 正常释放 {killed} 个, 兜底强杀 {leftover} 个; {how or '无异常'}", flush=True)

    @staticmethod
    def _force_kill_leftover():
        import subprocess
        import signal
        def pids():
            try:
                out = subprocess.run(["pgrep", "-f", "ray::RayWorker"],
                                     capture_output=True, text=True, timeout=10).stdout
                return [p for p in out.split() if p.isdigit()]
            except Exception:
                return []
        n = 0
        for pid in pids():
            try:
                os.kill(int(pid), signal.SIGTERM)
                n += 1
            except Exception:
                pass
        if n:
            import time
            time.sleep(5)
            for pid in pids():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
        return n


def _install_executor_tracker():
    """记下最新的 PromptExecutor 实例(用于清它的本地缓存)。"""
    import execution as _exec
    if getattr(_exec, "_h3_tracker_installed", False):
        return
    _exec._h3_tracker_installed = True
    _exec._h3_last_executor = None
    _orig_init = _exec.PromptExecutor.__init__

    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        # 进程生命周期内 executor 只有一个,直接持有强引用(不会泄漏)
        _exec._h3_last_executor = self

    _exec.PromptExecutor.__init__ = _patched_init


def _register():
    if not ENABLED:
        print("[h3-vram-release] 未启用(H3_VRAM_AUTORELEASE 未设置)—— 本插件休眠,不影响四卡模式", flush=True)
        return
    from comfy_execution.cache_provider import register_cache_provider
    register_cache_provider(H3VramReleaseProvider())
    print("[h3-vram-release] 已启用:每次跑完自动释放 Ray 显存并失效 RayLight 缓存", flush=True)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_install_executor_tracker()
_register()