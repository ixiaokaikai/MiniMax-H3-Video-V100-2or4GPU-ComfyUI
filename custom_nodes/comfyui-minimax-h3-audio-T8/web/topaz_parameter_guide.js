import { app } from "../../scripts/app.js";

const NODE_TYPE = "MiniMaxH3TopazVideoEXPT8";
const GUIDE_WIDGET = "topaz_parameter_reference";
const GUIDE_HEIGHT = 620;
const MIN_NODE_WIDTH = 760;
const GUIDE_TEXT = String.raw`【先看这里】
本说明只作参考，不参与执行。普通用户优先使用“参数模式=model_defaults”；节点会自动输出 H.264 MP4，不要再接“保存视频”。Topaz 软件无需保持打开。

【输入与基础设置】
topaz_runtime：连接“Official Topaz · 环境检查”节点的绿色输出。
source_video：连接未裁剪、未截段的本地文件视频；帧批次或编辑后视频请先保存成文件再加载。
高清模型 model_id：iris-3=真人脸/低清人像；prob-4=通用且手工参数最全；rhea-1=中等质量细节恢复；nyx-3/nxf-1/nxl-1=强降噪；ahq-12/amq-13/alq-13=高/中/低质量 Artemis；thf-4=忠实；thd-3=细节；ghq-5=高质量实拍/CG；gcg-5=CG/锯齿/摩尔纹；ganim-1=动画。
scale：1x 只增强不放大；2x 常用；4x 适合小图，但更慢且更占显存。示例：720p→1440p 选 2x。
vram_fraction：Topaz 可使用的显存比例。默认 0.80；显存紧张可试 0.60；不是预留显存门槛。
高级参数JSON parameters_json：通常保持 {}。它会最后覆盖同名参数，例如 {"noise":0.3,"details":0.5}；这不是提示词框。
size_mode：scale 使用倍率；target_dimensions 使用指定宽高并忽略 scale。
target_width / target_height：仅 target_dimensions 生效，须同时填写偶数、保持原比例且限制在原尺寸的 1–4 倍。示例：1920 × 1080。
output_directory：留空写入 ComfyUI/output/MiniMaxH3-Topaz；示例：D:\TopazOutput。不要填写磁盘根目录。
custom_model_id：通常留空；仅当正式安装出现列表尚未收录的新模型时填写，例如新的官方 model-id，并覆盖上方 model_id。
高级输出策略 output_profile：delivery_h264=默认 8-bit H.264 MP4；delivery_hevc_main10=明确需要 10-bit 时使用；lossless_master=审计母版，文件很大。HDR 仍须先转 SDR。

【参数模式 parameter_mode】
model_defaults：使用模型默认值，下面的手工滑块全部忽略。
auto_estimate：由 Topaz 抽帧分析；只读取“自动分析帧数 auto_estimate_frames”。默认 8 帧，复杂长片可试 12–20 帧，但会增加启动时间。
manual：读取下面的独立参数。不同模型支持项不同，不支持的恢复项会忽略，并写入 report_json.parameter_audit。

【manual 手工参数】
抗锯齿(-) / 去模糊(+) preblur：-1～1。负值偏抗锯齿/摩尔纹，正值偏去模糊。示例：-0.20 轻度抗锯齿；+0.20 轻度去模糊。
降噪 noise：-1～1。示例：0.20 轻度、0.45 中度；过高会蜡化皮肤。
恢复细节 details：-1～1。示例：0.25 自然、0.50 较强；过高可能生成假纹理。
去光晕 halo：-1～1。示例：0.20 轻度、0.50 明显边缘光晕；过高会损失边缘。
锐化 sharpen：-1～1，内部映射 Topaz blur 参数。示例：0.15 轻锐、0.35 较锐；高值可能白边。
去压缩伪影 compression：-1～1。示例：0.25 轻度网压、0.55 明显块效应。
输入预加噪 add_noise：0～0.10。默认 0；示例：0.01 可帮助过度平滑素材保留纹理。
输出颗粒 grain：0～1。默认 0；示例：0.10 轻颗粒、0.20 明显胶片感。
颗粒尺寸 grain_size：0～5，仅 grain>0 时有意义。示例：1.0 细颗粒、2.0 较粗。
模型色彩校正 keep_color：默认 true，尽量维持模型校正后的颜色；色彩对比测试可设 false。
混合原片 input_blend：0～1。0=完全增强结果，1=完全原片。示例：0.10～0.20 可减弱过度锐化或蜡感。
额外推理实例 engine_instances：0 最稳；1 仅在显存充足时尝试。本机 16GB 不建议 2 或 3，收益通常小而显存风险高。
Topaz GPU序号 gpu_device_index：单卡保持 0；多卡才选择其他正式 Topaz GPU 序号。

【五个可直接照抄的示例】
1. 真人脸一键 2x：iris-3；2x；model_defaults；JSON={}；vram_fraction=0.80。
2. 通用自动分析：prob-4；2x；auto_estimate；自动分析帧数=8；JSON={}。
3. 有噪点和压缩的人像：prob-4；2x；manual；preblur=0.10，noise=0.45，details=0.35，halo=0.25，sharpen=0.20，compression=0.50，add_noise=0，grain=0，input_blend=0.10。
4. 动画/卡通 2x：ganim-1；2x；model_defaults；JSON={}。
5. 精确输出 1080p：prob-4；size_mode=target_dimensions；target_width=1920；target_height=1080；model_defaults；JSON={}。

建议从默认值或示例开始，每次只改一两项。数值越高不代表越好；重点检查人脸蜡化、假细节、边缘白边、颜色漂移和颗粒。`;

function setStyle(style, name, value) {
    if (typeof style?.setProperty === "function") style.setProperty(name, value);
    else if (style) style[name] = value;
}

function createGuide(node) {
    let widget = node?.widgets?.find((item) => item?.name === GUIDE_WIDGET ||
        item?.name === "manual_parameter_guide");
    if (!widget && typeof document !== "undefined" && typeof node?.addDOMWidget === "function") {
        const input = document.createElement("textarea");
        input.value = GUIDE_TEXT;
        widget = node.addDOMWidget(GUIDE_WIDGET, "topaz-parameter-reference", input, {
            serialize: false,
            hideOnZoom: false,
        });
        widget.inputEl ??= input;
    }
    return widget;
}

function formatGuide(node) {
    const widget = createGuide(node);
    if (!widget) return;

    // A former development build serialized this help as an optional input.
    // Reuse that final-position widget when present, but never execute its value.
    widget.value = GUIDE_TEXT;
    widget.options ??= {};
    widget.options.serialize = false;
    widget.options.getMinHeight = () => GUIDE_HEIGHT;
    widget.options.getHeight = () => GUIDE_HEIGHT;
    widget.options.getMaxHeight = () => GUIDE_HEIGHT;

    const input = widget.inputEl;
    if (input) {
        input.value = GUIDE_TEXT;
        input.readOnly = true;
        input.spellcheck = false;
        input.title = "全参数、推荐范围与示例；只读，不参与Topaz执行";
        setStyle(input.style, "font-size", "14px");
        setStyle(input.style, "line-height", "1.55");
        setStyle(input.style, "padding", "12px");
        setStyle(input.style, "white-space", "pre-wrap");
        setStyle(input.style, "overflow-y", "auto");
        setStyle(input.style, "resize", "vertical");
    }

    const computed = typeof node.computeSize === "function" ? node.computeSize() : null;
    const current = Array.isArray(node.size) ? node.size : [0, 0];
    const width = Math.max(MIN_NODE_WIDTH, current[0] || 0, computed?.[0] || 0);
    const height = Math.max(current[1] || 0, computed?.[1] || GUIDE_HEIGHT);
    if (typeof node.setSize === "function") node.setSize([width, height]);
    else node.size = [width, height];
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "minimax-h3-audio-t8.topaz-parameter-guide",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            formatGuide(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            formatGuide(this);
            return result;
        };
    },
});
