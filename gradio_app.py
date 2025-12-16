"""Gradio demo for the two-stage inference pipeline.

This app wraps the existing command-line inference utilities so that users can
upload an input image and bounding boxes, run the generation pipeline, preview
one of the generated GLB meshes online, and download all generated artifacts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional
import shutil

import gradio as gr

from inference import (
    DataOverrides,
    StageSettings,
    _instantiate_stage1,
    _instantiate_stage2,
    _prepare_raw_batch,
    _resolve_device,
    _stage1_inference,
    _stage2_inference,
    _to_device_dtype,
)


def _parse_decode_ids(raw_value: str) -> Optional[List[int]]:
    if raw_value is None or raw_value.strip() == "":
        return None
    parts = [p.strip() for p in raw_value.split(",") if p.strip()]
    decoded_ids: List[int] = []
    for part in parts:
        try:
            decoded_ids.append(int(part))
        except ValueError:
            continue
    return decoded_ids or None


@lru_cache(maxsize=2)
def _load_pipelines(
    stage1_ckpt: str,
    stage2_ckpt: str,
    device_str: str,
    stage1_output_dir: str,
    stage2_output_dir: str,
    skip_stage2: bool,
):
    device = _resolve_device(device_str)
    stage1_settings = StageSettings(transformer_ckpt=stage1_ckpt or None)
    stage2_settings = StageSettings(
        config_key="3dmaster_part_s2", transformer_ckpt=stage2_ckpt or None
    )

    _, stage1_pipeline = _instantiate_stage1(
        stage1_settings, DataOverrides(), device
    )

    stage2_pipeline = None
    if not skip_stage2:
        stage2_pipeline = _instantiate_stage2(
            stage2_settings, Path(stage1_output_dir), Path(stage2_output_dir), device
        )

    return stage1_pipeline, stage2_pipeline, device


def run_inference(
    image_file,
    box_file,
    sample_id: str,
    stage1_ckpt: str,
    stage2_ckpt: str,
    output_root: str,
    decode_part_ids: str,
    skip_stage1: bool,
    skip_stage2: bool,
    device: str,
):
    if image_file is None or box_file is None:
        return None, None, "请上传输入图片和对应的边界框 .npy 文件。"

    sample_id = sample_id or "sample"
    output_dir = Path(output_root or "./gradio_outputs").expanduser().resolve()
    stage1_output_dir = output_dir / "stage1"
    stage2_output_dir = output_dir / "stage2"
    stage1_output_dir.mkdir(parents=True, exist_ok=True)
    if not skip_stage2:
        stage2_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        stage1_pipeline, stage2_pipeline, device_obj = _load_pipelines(
            stage1_ckpt,
            stage2_ckpt,
            device,
            str(stage1_output_dir),
            str(stage2_output_dir),
            skip_stage2,
        )
    except Exception as exc:  # noqa: BLE001
        return None, None, f"模型加载失败: {exc}"

    try:
        batch = _prepare_raw_batch(
            Path(image_file.name), Path(box_file.name), sample_id, None
        )
        batch = _to_device_dtype(batch, device_obj, "bf16")
    except Exception as exc:  # noqa: BLE001
        return None, None, f"输入预处理失败: {exc}"

    log_lines = []
    if not skip_stage1:
        log_lines.append("运行第一阶段推理…")
        _stage1_inference(stage1_pipeline, batch, stage1_output_dir)
    else:
        log_lines.append("已跳过第一阶段推理。")

    decode_ids = _parse_decode_ids(decode_part_ids)
    model_path = None
    archive_path = None
    if stage2_pipeline is not None:
        log_lines.append("运行第二阶段推理…")
        _stage2_inference(
            stage2_pipeline,
            batch,
            stage2_output_dir,
            decode_part_id=decode_ids,
        )
        eval_dir = stage2_output_dir / sample_id / "eval_batches"
        glb_files = sorted(eval_dir.glob("*.glb"))
        if glb_files:
            model_path = str(glb_files[0])
            archive_base = output_dir / f"{sample_id}_results"
            archive_path = shutil.make_archive(str(archive_base), "zip", eval_dir)
            log_lines.append(
                f"生成完成，共输出 {len(glb_files)} 个 GLB 文件。"
            )
        else:
            log_lines.append("未找到生成的 GLB 文件，请检查日志。")
    else:
        log_lines.append("已跳过第二阶段推理，仅保留第一阶段输出。")
        archive_path = shutil.make_archive(
            str(output_dir / f"{sample_id}_stage1"), "zip", stage1_output_dir
        )

    status = "\n".join(log_lines)
    return model_path, archive_path, status


def build_demo():
    with gr.Blocks(title="PartVerse Inference") as demo:
        gr.Markdown(
            """
            ## PartVerse 推理 Demo
            上传条件图片与对应的 bounding box (`.npy`)，运行两阶段生成流程，在线查看 3D 结果并下载文件。
            """
        )

        with gr.Row():
            image_input = gr.Image(
                label="条件图片 (PNG)", type="filepath", image_mode="RGBA"
            )
            box_input = gr.File(label="边界框 (.npy)")

        with gr.Row():
            sample_id = gr.Textbox(
                label="样本 ID", value="sample", info="用于标识输出文件夹"
            )
            output_root = gr.Textbox(
                label="输出目录", value="./gradio_outputs", info="保存推理结果的根目录"
            )

        with gr.Row():
            stage1_ckpt = gr.Textbox(
                label="Stage1 Transformer Checkpoint",
                placeholder="留空使用默认配置",
            )
            stage2_ckpt = gr.Textbox(
                label="Stage2 Transformer Checkpoint",
                placeholder="留空使用默认配置",
            )

        with gr.Row():
            decode_ids = gr.Textbox(
                label="可选：指定需要解码的 part id (逗号分隔)",
                placeholder="示例: 0,1,2",
            )
            device = gr.Dropdown(
                label="设备", choices=["cuda", "cpu"], value="cuda", allow_custom_value=True
            )

        with gr.Row():
            skip_stage1 = gr.Checkbox(label="跳过第一阶段", value=False)
            skip_stage2 = gr.Checkbox(label="跳过第二阶段", value=False)

        run_button = gr.Button("开始推理", variant="primary")

        model_viewer = gr.Model3D(label="生成的 3D 结果 (首个 GLB)")
        download = gr.File(label="下载生成文件 (ZIP)")
        status = gr.Textbox(label="状态", lines=6)

        run_button.click(
            run_inference,
            inputs=[
                image_input,
                box_input,
                sample_id,
                stage1_ckpt,
                stage2_ckpt,
                output_root,
                decode_ids,
                skip_stage1,
                skip_stage2,
                device,
            ],
            outputs=[model_viewer, download, status],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
