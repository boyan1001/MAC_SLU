# origin
# {"意图1": {"车载控制": [{"value": "车身控制", "name": "intent"}, {"name": "操作", "value": "打开"}, {"name": "对象", "value": "空调"}]}, "意图2": {"车载控制": [{"value": "车身控制", "name": "intent"}, {"name": "操作", "value": "打开"}, {"name": "对象", "value": "座椅"}, {"name": "对象功能", "value": "通风"}]}}
# {"意图1": {"音乐": [{"name": "操作", "value": "播放"}, {"name": "歌曲名", "value": "再也不会有人比我更爱你"}, {"value": "播放音乐", "name": "intent"}]}}
# {"意图1": {"音乐": [{"name": "操作", "value": "不想听"}, {"name": "对象", "value": "歌"}, {"value": "播放控制", "name": "intent"}]}}
# {"意图1": {"地图": [{"name": "操作", "value": "导航"}, {"name": "终点名称", "value": "北纬40度"}, {"value": "导航", "name": "intent"}]}}
# {"意图1": {"音乐": [{"name": "操作", "value": "放"}, {"name": "歌手名", "value": "苏鑫"}, {"name": "对象", "value": "歌"}, {"value": "播放音乐", "name": "intent"}]}, "意图2": {"车载控制": [{"value": "提供信息", "name": "intent"}, {"name": "操作", "value": "打开"}, {"name": "模式", "value": "内循环"}, {"value": "模式", "name": "调节内容"}]}}

# target
# [{"domain": "车载控制", "intent": "车身控制", "slots": {"操作": "打开", "对象": "空调"}}, {"domain": "车载控制", "intent": "车身控制", "slots": {"操作": "打开", "对象": "座椅", "对象功能": "通风"}}]
# [{"domain": "音乐", "intent": "播放音乐", "slots": {"操作": "播放", "歌曲名": "再也不会有人比我更爱你"}}]
# [{"domain": "音乐", "intent": "播放控制", "slots": {"操作": "不想听", "对象": "歌"}}]
# [{"domain": "地图", "intent": "导航", "slots": {"操作": "导航", "终点名称": "北纬40度"}}]
# [{"domain": "音乐", "intent": "播放音乐", "slots": {"操作": "放", "歌手名": "苏鑫", "对象": "歌"}}, {"domain": "车载控制", "intent": "提供信息", "slots": {"操作": "打开", "模式": "内循环", "调节内容": "模式"}}]

from typing import Any, Dict, List

def transform_semantics_to_standard(raw_semantics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    將原始訓練集的 semantics 格式轉換為系統提示詞要求的標準格式。
    """
    standard_list = []
    
    # 遍歷 意图1, 意图2...
    for intent_key, domains in raw_semantics.items():
        # 遍歷 領域 (如 音乐, 地图)
        for domain_name, slots_list in domains.items():
            new_frame = {
                "domain": domain_name,
                "intent": "",
                "slots": {}
            }
            
            # 提取 intent 欄位並重組 slots
            actual_slots = {}
            for item in slots_list:
                if item.get("name") == "intent":
                    new_frame["intent"] = item.get("value", "")
                else:
                    # 將 {"name": "歌手名", "value": "周杰倫"} 轉為 "歌手名": "周杰倫"
                    actual_slots[item["name"]] = item["value"]
            
            new_frame["slots"] = actual_slots
            standard_list.append(new_frame)
            
    return standard_list

# 測試範例
if __name__ == "__main__":
    raw_data_list = [
        {"意图1": {"车载控制": [{"value": "车身控制", "name": "intent"}, {"name": "操作", "value": "打开"}, {"name": "对象", "value": "空调"}]}, "意图2": {"车载控制": [{"value": "车身控制", "name": "intent"}, {"name": "操作", "value": "打开"}, {"name": "对象", "value": "座椅"}, {"name": "对象功能", "value": "通风"}]}},
        {"意图1": {"音乐": [{"name": "操作", "value": "播放"}, {"name": "歌曲名", "value": "再也不会有人比我更爱你"}, {"value": "播放音乐", "name": "intent"}]}},
        {"意图1": {"音乐": [{"name": "操作", "value": "不想听"}, {"name": "对象", "value": "歌"}, {"value": "播放控制", "name": "intent"}]}},
        {"意图1": {"地图": [{"name": "操作", "value": "导航"}, {"name": "终点名称", "value": "北纬40度"}, {"value": "导航", "name": "intent"}]}},
        {"意图1": {"音乐": [{"name": "操作", "value": "放"}, {"name": "歌手名", "value": "苏鑫"}, {"name": "对象", "value": "歌"}, {"value": "播放音乐", "name": "intent"}]}, "意图2": {"车载控制": [{"value": "提供信息", "name": "intent"}, {"name": "操作", "value": "打开"}, {"name": "模式", "value": "内循环"}, {"value": "模式", "name": "调节内容"}]}},
    ]

    for raw_data in raw_data_list:
        standard_output = transform_semantics_to_standard(raw_data)
        print(standard_output)