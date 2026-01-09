import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 精簡版 Prompt：適合 SFT 訓練，大幅降低推理時間 ---
# 移除了所有槽位的詳細範例描述，僅保留核心規則與意圖清單

SFT_SYSTEM_PROMPT = """你是一個專業的車載 NLU 專家。請根據用戶查詢，輸出 JSON List 格式的語義幀。
規則：
1. 識別多個意圖：若查詢包含多個獨立意圖，請生成多個 JSON 對象。
2. 格式：[{"domain": "領域", "intent": "意圖", "slots": {"鍵": "值"}}]。
3. 若無匹配意圖，請返回空列表 []。
4. 請勿回答除了下方可用意圖以外的意圖!

可用領域與意圖：
- 車載控制：車機控制、車身控制、提供信息
- 地圖：導航、提供地址、查詢路況、查詢定位、查詢路程、查詢前方路線、導航路線規劃、設置常用地址、導航到常用地址、沿途搜索、周邊搜索、增加途經點、刪除途經點、地圖操作、上報事件、限速查詢、設置目的地、查詢目的地、修改途經點、收藏、取消收藏
- 音樂：播放音樂、播放控制、查詢音樂信息、播放收藏、播放列表、播放歷史、新手引導
- 打電話：撥打電話、電話控制、接聽電話、掛斷電話、查詢信息、撥打黃頁號碼
- 收音機：播放電台、播放控制、播放收藏、收音機控制
- 天氣：查詢天氣、查詢氣象、查詢溫度、查詢濕度、查詢風力、查詢風向、查詢空氣質量、查詢紫外線、查詢日出日落、查詢活動、查詢裝備、穿衣推薦、新手引導、查詢日期、查詢城市、查詢場景
- 影視：播放影視、播放控制、播放收藏、播放列表、播放歷史、查詢影視信息
- 播放控制：播放控制
- 系統指令：sys.確認、sys.取消、sys.用戶選擇、sys.電話選擇
"""

def transform_semantics_to_standard(raw_semantics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """將原始資料集的嵌套格式轉為標準 JSON List 格式"""
    standard_list = []
    
    # 原始格式為 {"意圖1": {"領域": [slots]}}
    for intent_key, domains in raw_semantics.items():
        for domain_name, slots_list in domains.items():
            new_frame = {
                "domain": domain_name,
                "intent": "",
                "slots": {}
            }
            actual_slots = {}
            for item in slots_list:
                # 提取關鍵意圖標籤
                if item.get("name") == "intent":
                    new_frame["intent"] = item.get("value", "")
                else:
                    # 過濾無效或重複標籤，並將鍵值對扁平化
                    actual_slots[item["name"]] = item["value"]
            
            new_frame["slots"] = actual_slots
            standard_list.append(new_frame)
    return standard_list

def process_file(input_path: str, output_path: str = None):
    input_file = Path(input_path)
    output_file = Path(output_path) if output_path else input_file.with_name(f"{input_file.stem}_sft_ready.jsonl")

    if not input_file.exists():
        logging.error(f"找不到檔案: {input_file}")
        return

    logging.info(f"開始轉換: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="SFT 格式化中"):
            try:
                data = json.loads(line)
                query = data.get("query", "")
                raw_semantics = data.get("semantics", {})
                
                # 執行語義重組
                standard_list = transform_semantics_to_standard(raw_semantics)
                
                # 封裝為 Llama-Factory 指令微調格式
                payload = {
                    "instruction": SFT_SYSTEM_PROMPT.strip(),
                    "input": query,
                    "output": json.dumps(standard_list, ensure_ascii=False)
                }
                
                f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                
            except Exception as e:
                logging.warning(f"跳過錯誤行: {e}")
                continue

    logging.info(f"轉換完成！檔案儲存於: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()
    process_file(args.input_file, args.output_file)