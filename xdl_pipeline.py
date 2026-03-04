import os
import json
import time
from openai import OpenAI
from xdl_validator import XDLValidator
from config import Config
import xml.etree.ElementTree as ET

# === 辅助函数 ===
def load_xdl_spec(file_path="XDL_description_build.txt"):
    """加载外部 XDL 规范文件"""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Spec file '{file_path}' not found. Using default internal rules.")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading spec file: {e}")
        return None

def extract_reagents_from_xdl(xdl_string):
    """提取 Reagent 信息，保持键值一致"""
    try:
        clean_xdl = xdl_string.replace("```xml", "").replace("```", "").strip()
        if "<?xml" not in clean_xdl:
            clean_xdl = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + clean_xdl
        root = ET.fromstring(clean_xdl)
        reagents_node = root.find(".//Reagents")
        reagents_info = {}
        if reagents_node is not None:
            for r in reagents_node:
                name = r.get("name")
                if name: reagents_info[name] = r.get("formula", name)
        return reagents_info
    except Exception as e:
        print(f"Error parsing XDL: {e}")
        return {}

def extract_chem_analysis(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("chem_analysis", "")

# === 配置 ===
MODEL_NAME = "gpt-4.1" # 建议使用 GPT-4 级别模型
config = Config()
client = OpenAI(api_key=config.openai_api_key, base_url=config.base_url)

def generate_database_consistent(chem_text, xdl_reagents_dict):
    """
    基于 XDL 中已经确定的 Reagent Name 生成属性，确保一一对应。
    """
    # 将 XDL 中确定的物质列表转为字符串，提示 LLM 必须用这些名字
    reagent_list_str = json.dumps(xdl_reagents_dict, indent=2)
    
    prompt = f"""
    You are a chemical data engineer.
    
    Context:
    1. We have an experiment described as: "{chem_text}"
    2. We have ALREADY generated the execution script (XDL), which uses the following specific reagent names and formulas:
    {reagent_list_str}
    
    Task:
    Generate the `substances` and `reactions` database JSON.
    
    CRITICAL RULES:
    1. For `substances`: You MUST include keys for EXACTLY the names listed in the provided reagent list. Do not change spelling or capitalization.
    2. If the reaction produces NEW substances (products) not in the list, you should add them to `substances` as well.
    3. For `reactions`: Write the balanced equation using the formulas consistent with the list.
    
    Output JSON format:
    {{
        "substances": {{
            "Exact_Name_From_List": {{ "formula": "...", "molar_mass": float, "state": "s/l/g/aq", "color_rgb": [r,g,b,a], "description": "..." }},
            "New_Product_Name": {{ ... }}
        }},
        "reactions": [
            {{ "equation": "balanced string", "phenomena": "..." }}
        ]
    }}
    """
    
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(resp.choices[0].message.content)

def generate_xdl_unlimited(chem_text, max_retries=5):
    """
    XDL 生成器 (基于外部 TXT 规范)
    """
    
    # 1. 加载外部规范文本 
    xdl_spec_content = load_xdl_spec("XDL_description_build.txt")
    
    # 2. 构建 Prompt
    # 如果文件加载失败，可以使用一个简化的兜底规则，或者直接报错
    if not xdl_spec_content:
        xdl_spec_content = "Critical: Use standard XDL format with <Synthesis> wrapper."

    # === System Prompt ===
    # 将 txt 内容作为 "OFFICIAL SPECIFICATION" 注入
    system_prompt = f"""
    You are an expert in chemical XDL synthesis file generation (v3.0).
    
    [OFFICIAL SPECIFICATION - STRICTLY FOLLOW]
    {xdl_spec_content}
    
    [ADDITIONAL CRITICAL RULES]
    1. **Structure Integrity**: You MUST wrap <Hardware>, <Reagents>, and <Procedure> inside a <Synthesis> tag, as shown in the Example section of the spec.
    2. **Anti-Hallucination**: 
       - Do NOT invent tags like <MeasureObservation>, <Observe>, or <Check>. 
       - Use <Wait time="..."/> for qualitative observations (color change, bubbles).
    3. **Action constraints**:
       - <Add>: Strictly follow the 'state' rule (Liquid->volume, Solid->mass).
       - <Transfer>: Only for liquids between vessels.
    """

    user_prompt = f"""
    Convert this description into XDL v3.0 XML:
    "{chem_text}"
    
    Requirements:
    - Parse the 'goal' carefully.
    - Ensure every vessel used in Procedure is defined in Hardware.
    - Output pure XML.
    """

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        print(f"\n[Attempt {attempt+1}/{max_retries}] Generating XDL...")
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=history, temperature=0.2
            )
            raw_content = resp.choices[0].message.content
            
            clean_content = raw_content.replace("```xml", "").replace("```", "").strip()
            if "<?xml" not in clean_content:
                clean_content = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + clean_content

            # === 调用验证器 ===
            errors = XDLValidator.verify_xdl(clean_content)
            
            if not errors:
                print("✅ XDL Verification Passed!")
                return clean_content
            
            # 错误反馈
            error_msg = "Validation Failed:\n" + "\n".join([f"- {e}" for e in errors])
            print(f"⚠️ Errors found: {len(errors)}. Retrying...")
            
            history.append({"role": "assistant", "content": clean_content})
            history.append({"role": "user", "content": error_msg + "\nPlease fix strictly according to the [OFFICIAL SPECIFICATION] provided above."})
        
        except Exception as e:
            print(f"API Error: {e}")
            time.sleep(1)

    return None

def main():
    # 请修改为实际的输入 JSON 路径
    target_file = "D:\\postgraduate\\多模态大模型_化学实验\\output\\json_results\\1 大理石与稀盐酸反应_sample_chem_train_0.json"
    
    if not os.path.exists(target_file):
        print(f"Input file not found: {target_file}")
        # 创建一个 dummy 文件方便测试
        dummy_data = {"chem_analysis": "Titration of Unknown HCl with 0.1M NaOH using Phenolphthalein."}
        with open("sample.json", "w") as f: json.dump(dummy_data, f)
        target_file = "sample.json"
        print("Created dummy file 'sample.json' for testing.")

    print(f"1. Reading analysis from {target_file}...")
    chem_desc = extract_chem_analysis(target_file)
    
    print("\n2. Generating XDL (Source of Truth)...")
    # 此时生成器会读取目录下的 XDL_description_build.txt
    final_xdl = generate_xdl_unlimited(chem_desc)

    if not final_xdl:
        print("❌ Failed to generate XDL.")
        return

    print("\n3. Extracting Reagents...")
    xdl_reagents = extract_reagents_from_xdl(final_xdl)
    
    print("\n4. Generating Database...")
    db_data = generate_database_consistent(chem_desc, xdl_reagents)
    
    # 保存
    if not os.path.exists("database"): os.makedirs("database")
    with open("database/substances.json", "w", encoding="utf-8") as f:
        json.dump(db_data.get("substances"), f, indent=2)
    with open("database/reactions.json", "w", encoding="utf-8") as f:
        json.dump(db_data.get("reactions"), f, indent=2)

    if not os.path.exists("experiments"): os.makedirs("experiments")
    with open("experiments/auto_generated.xdl", "w", encoding="utf-8") as f:
        f.write(final_xdl)
    
    print(f"\n✅ Pipeline Complete. XDL and DB generated.")

if __name__ == "__main__":
    main()