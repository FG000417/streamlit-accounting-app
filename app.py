import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter 
import jieba
import sys
import os # 用於檢查檔案是否存在

# --- 步驟零：配置與數據準備 ---

# 🚨 請將 'filename.xlsx' 替換成您的實際檔案路徑與名稱
DATA_FILE = r'會計歷史資料.xlsx' 

# 🚨 配置 K-近鄰的 K 值 (可以調整，通常是 3 或 5)
N_NEIGHBORS = 5

# 🚨 2. 費用代號與科目對應表 (請務必補齊所有 001~033 的項目)
FEE_MAPPING = {
    '001': {'Name': '其他費用', 'Code': '6135'},
    '002': {'Name': '資產設備採購', 'Code': '6137'},
    '003': {'Name': '雜項', 'Code': '612901'},
    '004': {'Name': '郵資', 'Code': '611602'},
    '005': {'Name': '加油油資', 'Code': '6140'},
    '006': {'Name': '文具用品', 'Code': '6113'},
    '007': {'Name': '寄快遞運費', 'Code': '6115'},
    '010': {'Name': '辦公室清潔費', 'Code': '6135'},
    '011': {'Name': '各類租金費用', 'Code': '6112'},
    '012': {'Name': '雲端服務費', 'Code': '611604'},
    '013': {'Name': '代扣稅款繳款', 'Code': '225202'},
    '014': {'Name': '代扣補充保費繳款', 'Code': '225204'},
    '015': {'Name': '勞務人力費用等', 'Code': '6133'},
    '016': {'Name': '捐贈物資現金等', 'Code': '6122'},
    '017': {'Name': '預付結帳單', 'Code': '1266'},
    '018': {'Name': '暫付原創運費', 'Code': '2251'},
    '019': {'Name': '預付費用-其他', 'Code': '1272'},
    '020': {'Name': '計程車/交通車資', 'Code': '6140'},
    '021': {'Name': '公司相關稅捐', 'Code': '6135'},
    '022': {'Name': '廠商贈禮/用餐', 'Code': '6121'},
    '023': {'Name': '水電瓦斯費', 'Code': '6119'},
    '024': {'Name': '預付費用', 'Code': '1260'},
    '025': {'Name': '暫付款', 'Code': '1281'},
    '026': {'Name': '職工福利', 'Code': '611601'},
    '027': {'Name': '申報薪資類別', 'Code': '6111'},
    '028': {'Name': '押金', 'Code': '1583'},
    '029': {'Name': '電話/網路費', 'Code': '611601'},
    '030': {'Name': '保險費', 'Code': '6120'},
    '031': {'Name': '廣告相關費用', 'Code': '614107'},
    '032': {'Name': '代付費用', 'Code': '1282'},
    '033': {'Name': '返鄉補助費用', 'Code': '6140'},
}

# 3. 定義中文分詞函數
def chinese_segmentation(text):
    """將中文文本分詞，以空格連接"""
    if pd.isna(text):
        return ""
    # 這裡可以加入停用詞處理 (stop words)，但為了簡潔暫時省略
    return " ".join(jieba.cut(str(text)))

@st.cache_resource # 使用 Streamlit 緩存，確保模型只訓練一次
def train_and_prepare_model(file_path):
    """從 Excel 文件讀取數據並訓練模型"""
    if not os.path.exists(file_path):
        st.error(f"錯誤：找不到訓練數據文件: {file_path}。請確保檔案存在。")
        return None, None, None, None

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"錯誤：讀取 Excel 文件失敗。請確認檔案格式是否正確。{e}")
        return None, None, None, None

    # 數據清洗：移除空值並確保欄位名稱正確
    df = df.dropna(subset=['摘要', '科目編號'])
    df['科目編號'] = df['科目編號'].astype(str)
    
    Y = df['科目編號'] 
    df['摘要_分詞'] = df['摘要'].apply(chinese_segmentation)
    X = df['摘要_分詞']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric='cosine') 
    model.fit(X_vectorized, Y)

    # 建立反向對應表 (REVERSE_MAPPING)
    REVERSE_MAPPING = {}
    for option_code, details in FEE_MAPPING.items():
        account_code = details['Code']
        if account_code not in REVERSE_MAPPING:
            REVERSE_MAPPING[account_code] = []
        REVERSE_MAPPING[account_code].append(option_code)

    return model, vectorizer, df, REVERSE_MAPPING

# --- 步驟二：預測與輸出 ---

def predict_account_with_support(summary, model, vectorizer, history_df, reverse_mapping):
    """
    預測 科目編號，然後反查對應的 費用類別代號 和 名稱。
    """
    if not summary.strip():
        return {'Code': 'N/A', 'Name': '摘要為空', 'Account': 'N/A'}, []

    seg_summary = chinese_segmentation(summary)
    new_X_vectorized = vectorizer.transform([seg_summary])
    
    # 獲取距離最近的 N 個鄰居 (這裡是 科目編號)
    distances, indices = model.kneighbors(new_X_vectorized)
    
    # 取得這 N 個最相似鄰居的 科目編號
    nearest_codes = history_df.iloc[indices[0]]['科目編號'].tolist()
    
    # 計算 科目編號 的投票數
    code_counts = Counter(nearest_codes)
    
    # 彙整 Top K 推薦列表
    recommendations = []
    final_option_map = {} 

    for account_code, count in code_counts.most_common():
        # 反查這個 科目編號 應該推薦哪些 費用類別代號
        option_codes = reverse_mapping.get(account_code, ['未知代號'])
        
        # 由於一個科目編號可能對應多個費用類別，我們將它們全部顯示
        for option_code in option_codes:
            details = FEE_MAPPING.get(option_code, {'Name': '未知名稱', 'Code': '未知編號'})
            
            recommendations.append({
                '代號': option_code,
                '名稱': details['Name'],
                '科目編號': account_code,
                '支持度': f"{count}/{N_NEIGHBORS}",
                '支持比例': f"{(count / N_NEIGHBORS) * 100:.0f}%"
            })
            
            # 將第一個推薦作為主要結果 (只記錄一次)
            if not final_option_map:
                final_option_map = {'Code': option_code, 'Name': details['Name'], 'Account': account_code}

    return final_option_map, recommendations

# --- Streamlit 網頁介面主程式 ---

st.set_page_config(page_title="會計費用智能分類小工具", layout="centered")
st.title("🤖 費用申請智能分類輔助系統")
st.markdown("---")

# 1. 訓練模型 (使用 Streamlit 緩存，只運行一次)
model, vectorizer, history_df, reverse_mapping = train_and_prepare_model(DATA_FILE)

if model is not None:
    st.success(f"✅ 模型加載完成，基於 {len(history_df)} 筆歷史數據。")
    st.subheader("📝 輸入費用摘要")
    
    # 2. 創建輸入框
    user_input = st.text_area("請輸入發票或費用申請的摘要內容：", height=100)

    if st.button("🔍 開始預測科目") and user_input:
        with st.spinner('AI 正在計算最佳科目...'):
            main_option, recommendations = predict_account_with_support(user_input, model, vectorizer, history_df, reverse_mapping)

            # 3. 顯示主要推薦結果
            st.markdown("---")
            st.header(f"💰 主要推薦科目 (User Option)")
            
            st.info(f"**代號：{main_option['Code']} / 名稱：{main_option['Name']}**")
            st.subheader(f"拋轉會計科目編號: `{main_option['Account']}`")
            
            st.markdown("---")
            
            # 4. 顯示 Top K 詳細推薦
            st.subheader("💡 Top K 推薦明細 (信心度)")
            
            displayed_accounts = set()
            rec_data = []
            
            for rec in recommendations:
                if rec['科目編號'] not in displayed_accounts:
                    rec_data.append({
                        "推薦代號": rec['代號'],
                        "科目名稱": rec['名稱'],
                        "會計編號": rec['科目編號'],
                        "支持比例": rec['支持比例'],
                    })
                    displayed_accounts.add(rec['科目編號'])
            
            st.dataframe(pd.DataFrame(rec_data), hide_index=True)
            
            st.caption(f"（系統根據最相似的 {N_NEIGHBORS} 筆歷史數據計算支持度）")

    st.markdown("---")
    st.markdown("##### *請遵循摘要輸入優化指南，以獲得最高準確度。*")
