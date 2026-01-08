import argparse
import json
import logging
import base64
import os
import requests
import random

from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    # 如果用户不使用本地模式，这个库不是必需的
    OpenAI = None 

# 使用 tqdm 显示进度条
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        print("tqdm is not installed, progress bar will not be shown. "
              "Install it with: pip install tqdm")
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def call_local_api(
    api_base: str,
    model_name: str,
    audio_path: Path,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    """
    Use OpenAI-compatible local API to transcribe audio file.
    """
    
    # 1) Load OpenAI module
    if OpenAI is None:
        raise ImportError("OpenAI module is needed on calling local api: pip install openai")

    client = OpenAI(base_url=api_base, api_key="no-need")

    # 2) Get audio data
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()

    # 3) Call local API
    try:
        response = client.audio.transcriptions.create(
            model=model_name,
            file=audio_data,
            response_format="json",
            temperature=temperature,
            max_tokens=max_tokens,
            language="zh",
        )
        return response.get("text", "")
    except Exception as e:
        logging.error(f"Error calling local API for {audio_path}: {e}")
        return None
        
def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce Dataset with LLaMa-Factory Format")
    parser.add_argument(
        "--input-file", type=str, required=True, help="Input JSONL file path"
    )
    parser.add_argument(
        "--audio-dir", type=str, required=True, help="Input Audio_dir file path"
    )
    parser.add_argument(
        "--output-file", type=str, required=False, help="Output JSONL file path"
    )

    # --- 修改 ---: 增加'local'选项，并更新help说明
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "azure", "local"],
        help="API提供商: 'google'/'azure' (通过Dashscope), 或 'local' (本地OpenAI兼容接口)"
    )
    # --- 新增 ---: 为本地部署模型增加 --api-base 参数
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://0.0.0.0:12355/v1",
        help="本地LLM服务的API基地址 (仅当 --provider='local' 时使用)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM服务的API key。本地服务通常不需要，云服务则必需。"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="whisper-medium",
        help="要使用的模型名称 (例如: 'gemini-2.5-flash', 或本地部署的模型名)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="生成文本的温度。"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="模型生成的最大token数量。"
    )

    return parser

def process_file(args: argparse.Namespace):
    input_path = Path(args.input_file)
    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output_file) if args.output_file else input_path.parent / f"{input_path.stem}_llama_factory.jsonl"

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        return

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines, desc="Processing lines"):
            try:
                data = json.loads(line)
                item_id = data.get("id")
                
                if not item_id:
                    logging.warning("Missing 'id' in data, skipping line.")
                    continue
                
                audio_path = audio_dir / f"id_{item_id}.wav"
                if not audio_path.exists():
                    logging.warning(f"Audio file not found: {audio_path}, skipping line.")
                    continue

                text = call_local_api(
                    api_base=args.api_base,
                    model_name=args.model_name,
                    audio_path=audio_path,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                if text is None:
                    logging.warning(f"Failed to transcribe audio for id {item_id}, skipping line.")
                    continue

                data['query'] = text

                out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            except Exception as e:
                logging.error(f"Error processing line: {e}")
                continue
    
    logging.info(f"Processing complete. Output written to {output_path}")

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    process_file(args)

if __name__ == "__main__":
    main()

