import pandas as pd

def generate_encoding_mapping(excel_path: str = "drug_disease_data.xlsx") -> Dict:
    """
    从Excel数据集生成药物、疾病的编码映射（索引从0开始）
    :param excel_path: 数据集路径
    :return: encoding_mapping = {"drugs": {drug: idx}, "diseases": {disease: idx}}
    """
    # 读取Excel（需安装openpyxl）
    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    
    # 提取唯一药物和疾病（去重）
    unique_drugs = df["drug_name"].dropna().unique().tolist()
    unique_diseases = df["Disease"].dropna().unique().tolist()
    
    # 构建编码映射（药物→索引，疾病→索引）
    encoding_mapping = {
        "drugs": {drug: idx for idx, drug in enumerate(unique_drugs)},
        "diseases": {disease: idx for idx, disease in enumerate(unique_diseases)}
    }
    
    # 保存映射到JSON（后续可复用）
    import json
    with open("encoding_mapping.json", "w", encoding="utf-8") as f:
        json.dump(encoding_mapping, f, indent=2)
    
    print(f"生成编码映射：{len(unique_drugs)}种药物，{len(unique_diseases)}种疾病")
    return encoding_mapping

if __name__ == "__main__":
    generate_encoding_mapping()