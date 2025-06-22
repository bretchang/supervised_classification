import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, 
    load_digits, make_classification, fetch_20newsgroups
)
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 設置頁面配置
st.set_page_config(
    page_title="監督式學習-分類互動教學平台",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS樣式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .small-text {
        font-size: 0.7rem !important;
        line-height: 1.2 !important;
    }
</style>
""", unsafe_allow_html=True)

# 側邊欄導航
st.sidebar.title("🎯 課程導航")
page = st.sidebar.radio(
    "選擇學習模塊：",
    [
        "🏠 監督式學習概述",
        "📊 數據集探索", 
        "📈 邏輯回歸",
        "🎯 K近鄰分類",
        "🌳 決策樹分類",
        "🌲 隨機森林分類",
        "🚀 梯度提升分類",
        "🎯 支持向量機",
        "🧮 貝葉斯分類器",
        "🧠 神經網路分類",
        "📏 評價指標詳解",
        "🔄 交叉驗證與穩定性",
        "⚖️ 資料不平衡處理",
        "🔍 模型可解釋性",
        "🏆 模型綜合比較"
    ]
)

# 數據集選擇放在側邊欄
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 數據集選擇")
dataset_choice = st.sidebar.selectbox("選擇數據集：", [
    "鳶尾花分類", "紅酒分類", "乳癌診斷", "手寫數字識別", "人工數據集", "新聞分類"
])

# 數據集簡介
dataset_info = {
    "鳶尾花分類": "🌸 經典3分類問題，適合入門學習 (1KB)",
    "紅酒分類": "🍷 紅酒品種分類，適合特徵工程 (13KB)",
    "乳癌診斷": "🩺 二分類醫學診斷，實際應用場景 (32KB)", 
    "手寫數字識別": "🔢 10分類挑戰，圖像識別入門 (180KB)",
    "人工數據集": "🎲 可控制複雜度，教學實驗用 (可調)",
    "新聞分類": "📰 文本分類，自然語言處理 (1.2MB)"
}

st.sidebar.markdown("### 📝 數據集特點")
for dataset, description in dataset_info.items():
    if dataset == dataset_choice:
        st.sidebar.markdown(f'<div class="small-text">✅ <strong>{dataset}</strong>: {description}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<div class="small-text"><strong>{dataset}</strong>: {description}</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 作者信息")
st.sidebar.info("**This tutorial was made by CCChang18** 🚀")

# 初始化session_state
if 'artificial_params' not in st.session_state:
    st.session_state.artificial_params = {
        'n_samples': 1000,
        'n_features': 10,
        'n_informative': 5,
        'n_redundant': 2,
        'n_classes': 3,
        'class_sep': 1.0,
        'random_state': 42
    }

# 數據載入函數
@st.cache_data
def load_datasets():
    datasets = {}
    
    # 鳶尾花數據集
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['target_names'] = [iris.target_names[i] for i in iris.target]
    datasets['鳶尾花分類'] = iris_df
    
    # 紅酒數據集
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    wine_df['target_names'] = [wine.target_names[i] for i in wine.target]
    datasets['紅酒分類'] = wine_df
    
    # 乳癌診斷數據集
    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancer_df['target'] = cancer.target
    cancer_df['target_names'] = [cancer.target_names[i] for i in cancer.target]
    datasets['乳癌診斷'] = cancer_df
    
    # 手寫數字識別數據集
    digits = load_digits()
    digits_df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
    digits_df['target'] = digits.target
    digits_df['target_names'] = [str(i) for i in digits.target]
    datasets['手寫數字識別'] = digits_df
    
    # 人工數據集 - 使用session_state來存儲參數
    params = st.session_state.artificial_params
    X_artificial, y_artificial = make_classification(
        n_samples=params['n_samples'], 
        n_features=params['n_features'], 
        n_informative=params['n_informative'], 
        n_redundant=params['n_redundant'], 
        n_clusters_per_class=1, 
        n_classes=params['n_classes'],
        class_sep=params['class_sep'],
        random_state=params['random_state']
    )
    artificial_df = pd.DataFrame(X_artificial, columns=[f'feature_{i}' for i in range(X_artificial.shape[1])])
    artificial_df['target'] = y_artificial
    artificial_df['target_names'] = [f'Class_{i}' for i in y_artificial]
    datasets['人工數據集'] = artificial_df
    
    # 新聞分類數據集 (簡化版)
    try:
        # 只使用少數類別以減少複雜度
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        newsgroups = fetch_20newsgroups(subset='train', categories=categories, 
                                      remove=('headers', 'footers', 'quotes'))
        
        # 文本向量化 (簡化)
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_news = vectorizer.fit_transform(newsgroups.data).toarray()
        
        news_df = pd.DataFrame(X_news, columns=[f'word_{i}' for i in range(X_news.shape[1])])
        news_df['target'] = newsgroups.target
        news_df['target_names'] = [newsgroups.target_names[i] for i in newsgroups.target]
        datasets['新聞分類'] = news_df
    except:
        # 如果載入失敗，創建假數據
        X_fake, y_fake = make_classification(
            n_samples=500, n_features=20, n_classes=4, random_state=42
        )
        fake_df = pd.DataFrame(X_fake, columns=[f'word_{i}' for i in range(X_fake.shape[1])])
        fake_df['target'] = y_fake
        fake_df['target_names'] = [f'Topic_{i}' for i in y_fake]
        datasets['新聞分類'] = fake_df
    
    return datasets

all_datasets = load_datasets()

# 通用數據獲取函數
def get_current_data():
    try:
        current_dataset = all_datasets[dataset_choice]
        if current_dataset is None or len(current_dataset) == 0:
            st.error(f"❌ 數據集 '{dataset_choice}' 為空或無法加載")
            return pd.DataFrame(), pd.Series(dtype=int), []
        
        X = current_dataset.drop(['target', 'target_names'], axis=1)
        y = current_dataset['target']
        target_names = current_dataset['target_names'].unique()
        
        # 確保數據類型正確
        X = X.select_dtypes(include=[np.number])  # 只選擇數值型特徵
        
        if len(X.columns) == 0:
            st.error(f"❌ 數據集 '{dataset_choice}' 沒有數值型特徵")
            return pd.DataFrame(), pd.Series(dtype=int), []
            
        # 移除任何缺失值
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            st.error(f"❌ 數據集 '{dataset_choice}' 在移除缺失值後為空")
            return pd.DataFrame(), pd.Series(dtype=int), []
        
        return X, y, target_names
        
    except Exception as e:
        st.error(f"❌ 加載數據集 '{dataset_choice}' 時發生錯誤: {str(e)}")
        return pd.DataFrame(), pd.Series(dtype=int), []

# 頁面內容
if page == "🏠 監督式學習概述":
    st.markdown('<h1 class="main-header">監督式學習(Supervised Learning)-分類 互動教學平台</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🎯 什麼是分類？")
    
    st.markdown("""
    **分類(Classification)**是監督式學習的重要分支，目標是將數據點分配到預定義的類別中：
    
    1. **離散輸出**：預測結果是有限的類別標籤
    2. **決策邊界**：學習區分不同類別的邊界
    3. **概率估計**：多數算法可提供類別概率
    """)
    
    st.markdown("### 🔍 分類 vs 回歸")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **分類(Classification)**：預測**離散類別**
        - 垃圾郵件偵測 (垃圾/正常)
        - 疾病診斷 (患病/健康)
        - 圖像識別 (貓/狗/鳥)
        - 情感分析 (正面/負面/中性)
        """)
    
    with col2:
        st.markdown("""
        **回歸(Regression)**：預測**連續數值**
        - 房價預測
        - 溫度預測  
        - 股價預測
        - 銷售額預測
        """)
    
    st.markdown("### 📊 分類示例：決策邊界")
    
    # 創建分類示意圖
    np.random.seed(42)
    n_samples = 100
    
    # 生成兩類數據
    class_0_x = np.random.normal(2, 1, n_samples//2)
    class_0_y = np.random.normal(2, 1, n_samples//2)
    class_1_x = np.random.normal(4, 1, n_samples//2)
    class_1_y = np.random.normal(4, 1, n_samples//2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=class_0_x, y=class_0_y, mode='markers',
        name='類別 A', marker=dict(color='blue', size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=class_1_x, y=class_1_y, mode='markers',
        name='類別 B', marker=dict(color='red', size=8)
    ))
    
    # 添加決策邊界
    x_line = np.linspace(0, 6, 100)
    y_line = x_line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode='lines',
        name='決策邊界', line=dict(color='green', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="分類示例：尋找決策邊界區分不同類別",
        xaxis_title="特徵 1",
        yaxis_title="特徵 2",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## 📚 本課程學習內容")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔧 核心算法
        - 邏輯回歸
        - K近鄰分類
        - 決策樹分類
        - 隨機森林分類
        - 支持向量機
        - 貝葉斯分類器
        - 神經網路分類
        """)
    
    with col2:
        st.markdown("""
        ### 📏 評價指標
        - 準確率 (Accuracy)
        - 精確率 (Precision)
        - 召回率 (Recall)
        - F1分數
        - ROC-AUC
        - 混淆矩陣
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 特殊主題
        - 多分類問題
        - 資料不平衡處理
        - 模型可解釋性
        - 決策邊界視覺化
        - 特徵重要性分析
        """)
    
    st.markdown("## 🎯 學習目標")
    st.info("""
    通過本課程，您將能夠：
    1. 理解不同分類算法的原理和適用場景
    2. 掌握分類模型的評估方法
    3. 學會處理多分類和不平衡數據問題
    4. 進行模型比較和選擇
    5. 提升模型的可解釋性
    """)

elif page == "📊 數據集探索":
    st.markdown('<h1 class="main-header">📊 數據集探索</h1>', unsafe_allow_html=True)
    
    st.info("💡 您可以在左側選擇不同的數據集來探索其特性")
    
    # 人工數據集參數控制界面
    if dataset_choice == "人工數據集":
        st.markdown("## 🎛️ 人工數據集參數控制")
        st.info("💡 您可以調整人工數據集的參數來創建不同複雜度的分類問題")
        
        # 參數控制界面
        col1, col2 = st.columns(2)
        
        # 獲取當前參數
        current_params = st.session_state.artificial_params
        
        with col1:
            n_samples = st.slider("樣本數量：", 100, 2000, current_params['n_samples'], 100)
            n_features = st.slider("總特徵數：", 5, 20, current_params['n_features'])
            n_informative = st.slider("有用特徵數：", 2, n_features, min(current_params['n_informative'], n_features))
        
        with col2:
            n_redundant = st.slider("冗餘特徵數：", 0, n_features-n_informative, min(current_params['n_redundant'], n_features-n_informative))
            n_classes = st.slider("類別數量：", 2, 5, current_params['n_classes'])
            class_sep = st.slider("類別分離度：", 0.1, 3.0, current_params['class_sep'], 0.1)
        
        random_state = st.slider("隨機種子：", 1, 100, current_params['random_state'])
        
        # 更新按鈕
        if st.button("🔄 更新數據集", type="primary"):
            st.session_state.artificial_params = {
                'n_samples': n_samples,
                'n_features': n_features,
                'n_informative': n_informative,
                'n_redundant': n_redundant,
                'n_classes': n_classes,
                'class_sep': class_sep,
                'random_state': random_state
            }
            st.cache_data.clear()  # 清除緩存以重新生成數據
            st.rerun()
        
        # 顯示當前參數
        st.markdown("### 📋 當前參數配置")
        params_info = f"""
        - **樣本數量**: {current_params['n_samples']}
        - **總特徵數**: {current_params['n_features']}
        - **有用特徵數**: {current_params['n_informative']}
        - **冗餘特徵數**: {current_params['n_redundant']}
        - **類別數量**: {current_params['n_classes']}
        - **類別分離度**: {current_params['class_sep']}
        """
        st.info(params_info)
    
    # 獲取當前選擇的數據集
    current_dataset = all_datasets[dataset_choice]
    
    # 數據集信息映射 - 動態生成人工數據集信息
    artificial_params = st.session_state.artificial_params
    dataset_descriptions = {
        "鳶尾花分類": {
            "title": "🌸 鳶尾花品種分類數據集",
            "target_desc": "鳶尾花品種 (setosa, versicolor, virginica)",
            "source": "經典機器學習數據集",
            "features": {
                "sepal length": "花萼長度", "sepal width": "花萼寬度", 
                "petal length": "花瓣長度", "petal width": "花瓣寬度"
            },
            "color": "lightblue",
            "n_classes": 3
        },
        "紅酒分類": {
            "title": "🍷 紅酒品種分類數據集", 
            "target_desc": "紅酒品種類別 (0, 1, 2)",
            "source": "UCI機器學習庫",
            "features": {
                "alcohol": "酒精度", "malic_acid": "蘋果酸", 
                "ash": "灰分", "total_phenols": "總酚類"
            },
            "color": "lightcoral",
            "n_classes": 3
        },
        "乳癌診斷": {
            "title": "🩺 乳癌診斷數據集",
            "target_desc": "診斷結果 (良性/惡性)",
            "source": "威斯康辛大學醫院",
            "features": {
                "mean radius": "平均半徑", "mean texture": "平均紋理", 
                "mean perimeter": "平均周長", "mean area": "平均面積"
            },
            "color": "lightseagreen",
            "n_classes": 2
        },
        "手寫數字識別": {
            "title": "🔢 手寫數字識別數據集",
            "target_desc": "數字類別 (0-9)",
            "source": "scikit-learn內建數據集",
            "features": {
                "pixel_0": "像素0", "pixel_1": "像素1", 
                "pixel_n": "像素n", "...": "共64個像素特徵"
            },
            "color": "lightgoldenrodyellow",
            "n_classes": 10
        },
        "人工數據集": {
            "title": "🎲 人工生成分類數據集",
            "target_desc": f"人工類別 (Class_0 到 Class_{artificial_params['n_classes']-1})",
            "source": "sklearn.make_classification生成",
            "features": {
                "feature_0": "特徵0", "feature_1": "特徵1", 
                "feature_n": "特徵n", "...": f"共{artificial_params['n_features']}個特徵"
            },
            "color": "lightpink",
            "n_classes": artificial_params['n_classes']
        },
        "新聞分類": {
            "title": "📰 新聞主題分類數據集",
            "target_desc": "新聞主題類別",
            "source": "20newsgroups數據集",
            "features": {
                "word_0": "詞彙0", "word_1": "詞彙1", 
                "word_n": "詞彙n", "...": "TF-IDF特徵"
            },
            "color": "lightgreen",
            "n_classes": 4
        }
    }
    
    desc = dataset_descriptions[dataset_choice]
    st.markdown(f"## {desc['title']}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 數據集資訊")
        st.info(f"""
        - **樣本數量**: {len(current_dataset)} 個樣本
        - **特徵數量**: {len(current_dataset.columns)-2} 個特徵
        - **類別數量**: {desc['n_classes']} 個類別
        - **目標變數**: {desc['target_desc']}
        - **數據來源**: {desc['source']}
        """)
    
    with col2:
        st.markdown("### 🔬 主要特徵說明")
        for feature, description in desc['features'].items():
            st.markdown(f"- **{feature}**: {description}")
    
    # 類別分布
    st.markdown("### 📊 類別分布")
    class_counts = current_dataset['target'].value_counts().sort_index()
    class_names = current_dataset['target_names'].unique()
    
    fig = go.Figure(data=[
        go.Bar(x=class_names, y=class_counts.values, marker_color=desc['color'])
    ])
    fig.update_layout(
        title="各類別樣本數量分布",
        xaxis_title="類別",
        yaxis_title="樣本數量",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 檢查類別平衡性
    balance_ratio = class_counts.min() / class_counts.max()
    if balance_ratio < 0.5:
        st.warning(f"⚠️ 數據不平衡！最小類別與最大類別比例：{balance_ratio:.2f}")
    else:
        st.success(f"✅ 數據相對平衡，類別比例：{balance_ratio:.2f}")
    
    # 特徵分布可視化
    st.markdown("### 📈 特徵分布分析")
    
    X, y, target_names = get_current_data()
    if len(X) > 0:
        # 選擇前4個特徵進行可視化
        features_to_plot = X.columns[:4]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{col}" for col in features_to_plot]
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, feature in enumerate(features_to_plot):
            row = i // 2 + 1
            col = i % 2 + 1
            
            for class_idx, class_name in enumerate(target_names):
                class_data = X[y == class_idx][feature]
                fig.add_trace(
                    go.Histogram(
                        x=class_data, name=f'{class_name}', 
                        opacity=0.7, nbinsx=20,
                        marker_color=colors[class_idx % len(colors)],
                        showlegend=(i==0)  # 只在第一個子圖顯示圖例
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="各類別在不同特徵上的分布",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 邏輯回歸":
    st.markdown('<h1 class="main-header">📈 邏輯回歸 (Logistic Regression)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 📐 邏輯回歸公式")
    st.markdown("邏輯回歸使用Sigmoid函數將線性函數的輸出映射到0-1之間：")
    
    st.latex(r'''
    P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)}}
    ''')
    
    st.markdown("### 🎯 Sigmoid函數")
    st.latex(r'''
    \sigma(z) = \frac{1}{1 + e^{-z}}
    ''')
    
    st.markdown("### 📊 對數損失函數")
    st.latex(r'''
    J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(h_\beta(x_i)) + (1-y_i) \log(1-h_\beta(x_i))]
    ''')
    
    # 優缺點
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 📊 **提供概率輸出**：不只是分類，還有概率
        - 🔧 **簡單高效**：計算成本低
        - 🎯 **可解釋性強**：係數有明確含義
        - 📈 **無需特徵縮放**：對特徵範圍不敏感
        - 🛡️ **不假設數據分布**：非參數方法
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 📏 **假設線性關係**：決策邊界為線性
        - 🎯 **對離群值敏感**：極值影響較大
        - 🔗 **多重共線性問題**：特徵相關性高時不穩定
        - 📊 **需要大樣本**：小樣本時不穩定
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            solver = st.selectbox("求解器：", ["liblinear", "lbfgs", "newton-cg", "sag"])
            max_iter = st.slider("最大迭代次數：", 100, 2000, 1000, 100)
        
        with col2:
            test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
            random_seed = st.slider("隨機種子：", 1, 100, 42)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:4]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=random_seed
            )
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 建立模型
            model = LogisticRegression(
                solver=solver, max_iter=max_iter, random_state=random_seed
            )
            model.fit(X_train_scaled, y_train)
            
            # 預測
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            y_train_proba = model.predict_proba(X_train_scaled)
            y_test_proba = model.predict_proba(X_test_scaled)
            
            # 評估指標
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            # 計算其他指標
            if len(target_names) == 2:  # 二分類
                train_precision = precision_score(y_train, y_train_pred)
                test_precision = precision_score(y_test, y_test_pred)
                train_recall = recall_score(y_train, y_train_pred)
                test_recall = recall_score(y_test, y_test_pred)
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
            else:  # 多分類
                train_precision = precision_score(y_train, y_train_pred, average='weighted')
                test_precision = precision_score(y_test, y_test_pred, average='weighted')
                train_recall = recall_score(y_train, y_train_pred, average='weighted')
                test_recall = recall_score(y_test, y_test_pred, average='weighted')
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # 顯示結果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train-Data 準確率", f"{train_acc:.4f}")
            with col2:
                st.metric("Test-Data 準確率", f"{test_acc:.4f}")
            with col3:
                st.metric("Train-Data F1", f"{train_f1:.4f}")
            with col4:
                st.metric("Test-Data F1", f"{test_f1:.4f}")
            
            # 混淆矩陣
            st.markdown("### 📊 混淆矩陣")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_train = confusion_matrix(y_train, y_train_pred)
                fig = px.imshow(cm_train, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Training Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig = px.imshow(cm_test, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Test Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # 特徵係數分析
            st.markdown("### 📊 特徵係數分析")
            
            if len(target_names) == 2:  # 二分類
                coef_df = pd.DataFrame({
                    '特徵': selected_features,
                    '係數': model.coef_[0],
                    '絕對值': np.abs(model.coef_[0])
                }).sort_values('絕對值', ascending=False)
                
                fig = go.Figure(data=[
                    go.Bar(x=coef_df['特徵'], y=coef_df['係數'],
                           marker_color=['red' if x < 0 else 'blue' for x in coef_df['係數']])
                ])
                fig.update_layout(
                    title="邏輯回歸特徵係數",
                    xaxis_title="特徵",
                    yaxis_title="係數值",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig, use_container_width=True)
            else:  # 多分類
                st.info("多分類問題：每個類別都有一組係數")

elif page == "🎯 K近鄰分類":
    st.markdown('<h1 class="main-header">🎯 K近鄰分類 (K-Nearest Neighbors)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 📐 基本思想")
    st.markdown("KNN基於「相似的樣本應該有相似的標籤」的假設：")
    
    st.latex(r'''
    \hat{y} = \text{mode}(\{y_i : x_i \in N_k(x)\})
    ''')
    
    st.markdown("其中 $N_k(x)$ 是距離查詢點 $x$ 最近的 $k$ 個鄰居。")
    
    st.markdown("### 📏 距離度量")
    st.markdown("常用距離度量：")
    st.latex(r'''
    \begin{align}
    \text{歐氏距離：} d(x, y) &= \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2} \\
    \text{曼哈頓距離：} d(x, y) &= \sum_{i=1}^{n}|x_i - y_i| \\
    \text{閔可夫斯基距離：} d(x, y) &= (\sum_{i=1}^{n}|x_i - y_i|^p)^{1/p}
    \end{align}
    ''')
    
    # 優缺點
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🎯 **簡單直觀**：易於理解和實現
        - 🌊 **非參數方法**：不假設數據分布
        - 📊 **可處理多分類**：天然支持多類別
        - 🔄 **適應性強**：能適應局部數據模式
        - 🛠️ **無需訓練**：懶惰學習算法
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⏱️ **預測速度慢**：需要計算所有距離
        - 💿 **記憶體需求大**：需要存儲所有訓練數據
        - 📏 **對特徵縮放敏感**：不同尺度影響距離
        - 🎯 **對噪聲和離群值敏感**：近鄰可能是噪聲
        - 📈 **維度詛咒**：高維度時性能下降
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            n_neighbors = st.slider("鄰居數量 (k)：", 1, 20, 5)
            weights = st.selectbox("權重方式：", ["uniform", "distance"])
        
        with col2:
            metric = st.selectbox("距離度量：", ["euclidean", "manhattan", "minkowski"])
            test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:4]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42
            )
            
            # 標準化（KNN必須標準化）
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 建立模型
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
            )
            model.fit(X_train_scaled, y_train)
            
            # 預測
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # 評估指標
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            # 計算其他指標
            if len(target_names) == 2:  # 二分類
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
            else:  # 多分類
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # 顯示結果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train-Data 準確率", f"{train_acc:.4f}")
            with col2:
                st.metric("Test-Data 準確率", f"{test_acc:.4f}")
            with col3:
                st.metric("Train-Data F1", f"{train_f1:.4f}")
            with col4:
                st.metric("Test-Data F1", f"{test_f1:.4f}")
            
            # 過擬合檢測
            overfitting_gap = train_acc - test_acc
            if overfitting_gap > 0.1:
                st.warning(f"⚠️ 可能過擬合！準確率差距：{overfitting_gap:.4f}")
                st.markdown("**建議：** 增加k值或使用距離權重")
            elif overfitting_gap < -0.05:
                st.info("ℹ️ 模型可能欠擬合，可以減少k值")
            else:
                st.success("✅ 模型表現良好！")
            
            # 混淆矩陣
            st.markdown("### 📊 混淆矩陣")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_train = confusion_matrix(y_train, y_train_pred)
                fig = px.imshow(cm_train, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Training Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig = px.imshow(cm_test, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Test Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # K值影響分析
            st.markdown("### 📈 K值影響分析")
            
            k_range = range(1, min(21, len(X_train)))
            train_scores = []
            test_scores = []
            
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
                knn.fit(X_train_scaled, y_train)
                train_scores.append(knn.score(X_train_scaled, y_train))
                test_scores.append(knn.score(X_test_scaled, y_test))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(k_range), y=train_scores, mode='lines+markers', name='Training Accuracy'))
            fig.add_trace(go.Scatter(x=list(k_range), y=test_scores, mode='lines+markers', name='Test Accuracy'))
            fig.update_layout(
                title="不同K值對模型性能的影響",
                xaxis_title="K值",
                yaxis_title="準確率",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🌳 決策樹分類":
    st.markdown('<h1 class="main-header">🌳 決策樹分類 (Decision Tree Classification)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🌳 決策樹的基本概念")
    st.markdown("決策樹通過一系列if-else規則來進行分類：")
    
    st.latex(r'''
    \hat{y} = \text{決策路徑}(x_1, x_2, ..., x_n)
    ''')
    
    st.markdown("### 📊 不純度度量")
    st.markdown("常用的不純度度量：")
    
    st.latex(r'''
    \begin{align}
    \text{基尼不純度：} Gini(t) &= 1 - \sum_{i=1}^{c} p_i^2 \\
    \text{信息熵：} Entropy(t) &= -\sum_{i=1}^{c} p_i \log_2(p_i) \\
    \text{分類錯誤率：} Error(t) &= 1 - \max_i(p_i)
    \end{align}
    ''')
    
    st.markdown("### 🎯 信息增益")
    st.latex(r'''
    IG(T, A) = Entropy(T) - \sum_{v \in values(A)} \frac{|T_v|}{|T|} Entropy(T_v)
    ''')
    
    # 優缺點
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 📝 **易於理解和解釋**：樹狀結構直觀
        - 🔧 **不需要特徵標準化**：處理原始數據
        - 🌊 **能處理非線性關係**：複雜決策邊界
        - 🎯 **自動進行特徵選擇**：忽略不重要特徵
        - 📊 **能處理數值和類別特徵**：混合數據類型
        - 🛡️ **對離群值不敏感**：基於分割規則
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⚠️ **容易過擬合**：深度過大時
        - 🎲 **對訓練數據敏感**：小變化大影響
        - 📈 **偏向多值特徵**：選擇偏差
        - 📏 **難以捕捉線性關係**：需要多次分割
        - 🔄 **預測不穩定**：高方差問題
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            criterion = st.selectbox("分割標準：", ["gini", "entropy"])
            max_depth = st.slider("最大深度：", 1, 20, 5)
        
        with col2:
            min_samples_split = st.slider("最小分割樣本數：", 2, 20, 2)
            min_samples_leaf = st.slider("葉節點最小樣本數：", 1, 10, 1)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:4]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
            
            # 建立模型
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # 預測
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # 評估指標
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            if len(target_names) == 2:
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
            else:
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # 顯示結果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train-Data 準確率", f"{train_acc:.4f}")
            with col2:
                st.metric("Test-Data 準確率", f"{test_acc:.4f}")
            with col3:
                st.metric("Train-Data F1", f"{train_f1:.4f}")
            with col4:
                st.metric("Test-Data F1", f"{test_f1:.4f}")
            
            # 過擬合檢測
            overfitting_gap = train_acc - test_acc
            if overfitting_gap > 0.15:
                st.warning(f"⚠️ 決策樹過擬合嚴重！準確率差距：{overfitting_gap:.4f}")
            else:
                st.success("✅ 模型表現良好！")
            
            # 混淆矩陣
            st.markdown("### 📊 混淆矩陣")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_train = confusion_matrix(y_train, y_train_pred)
                fig = px.imshow(cm_train, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Training Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig = px.imshow(cm_test, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Test Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # 特徵重要性
            st.markdown("### 📊 特徵重要性分析")
            
            feature_importance = pd.DataFrame({
                '特徵': selected_features,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(x=feature_importance['特徵'], y=feature_importance['重要性'],
                       marker_color='green')
            ])
            fig.update_layout(
                title="決策樹特徵重要性",
                xaxis_title="特徵",
                yaxis_title="重要性",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 樹的結構信息
            st.info(f"**決策樹結構：** 實際深度={model.get_depth()}, 葉節點數={model.get_n_leaves()}")

elif page == "🌲 隨機森林分類":
    st.markdown('<h1 class="main-header">🌲 隨機森林分類 (Random Forest Classification)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🌲 隨機森林的基本概念")
    st.markdown("隨機森林是多個決策樹的集成，通過投票來決定最終分類：")
    
    st.latex(r'''
    \hat{y} = \text{mode}(\{T_1(x), T_2(x), ..., T_B(x)\})
    ''')
    
    st.markdown("### 🎲 Bootstrap + 隨機特徵")
    st.markdown("每棵樹使用：")
    st.markdown("1. **Bootstrap抽樣**：隨機抽取樣本（有放回）")
    st.markdown("2. **隨機特徵子集**：每次分割隨機選擇特徵")
    
    st.latex(r'''
    \text{每個分割點使用} \approx \sqrt{p} \text{個隨機特徵}
    ''')
    
    # 優缺點
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🛡️ **減少過擬合**：集成多棵樹
        - 📊 **提供特徵重要性**：量化特徵貢獻
        - 💾 **處理大數據集**：高效算法
        - 🎯 **對離群值穩健**：多數決定
        - ⚡ **可並行訓練**：獨立建樹
        - 📈 **提供OOB估計**：內建驗證
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 🔍 **模型不可解釋**：黑盒性質
        - 💿 **記憶體需求大**：存儲多棵樹
        - ⏱️ **預測時間較長**：多個模型預測
        - 📉 **可能在噪聲數據上過擬合**：學習噪聲
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("樹的數量：", 10, 200, 100, 10)
            max_depth = st.slider("最大深度：", 3, 20, 10)
        
        with col2:
            min_samples_split = st.slider("最小分割樣本數：", 2, 20, 2)
            test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42
            )
            
            # 建立模型
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # 預測
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_proba = model.predict_proba(X_train)
            y_test_proba = model.predict_proba(X_test)
            
            # 評估指標
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            if len(target_names) == 2:
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
            else:
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # 顯示結果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train-Data 準確率", f"{train_acc:.4f}")
            with col2:
                st.metric("Test-Data 準確率", f"{test_acc:.4f}")
            with col3:
                st.metric("Train-Data F1", f"{train_f1:.4f}")
            with col4:
                st.metric("Test-Data F1", f"{test_f1:.4f}")
            
            # 穩定性分析
            stability_gap = train_acc - test_acc
            if stability_gap > 0.1:
                st.warning(f"⚠️ 模型可能輕微過擬合，準確率差距：{stability_gap:.4f}")
            else:
                st.success("✅ 隨機森林模型穩定性良好！")
            
            # 混淆矩陣
            st.markdown("### 📊 混淆矩陣")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_train = confusion_matrix(y_train, y_train_pred)
                fig = px.imshow(cm_train, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Training Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig = px.imshow(cm_test, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Test Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # 特徵重要性分析
            st.markdown("### 📊 特徵重要性分析")
            
            feature_importance = pd.DataFrame({
                '特徵': selected_features,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(x=feature_importance['特徵'], y=feature_importance['重要性'],
                       marker_color='forestgreen')
            ])
            fig.update_layout(
                title=f"隨機森林特徵重要性 ({n_estimators}棵樹)",
                xaxis_title="特徵",
                yaxis_title="重要性",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 預測概率分布
            if len(target_names) <= 5:  # 只在類別不太多時顯示
                st.markdown("### 📈 預測概率分布")
                
                fig = go.Figure()
                for i, class_name in enumerate(target_names):
                    fig.add_trace(go.Histogram(
                        x=y_test_proba[:, i],
                        name=f'{class_name}',
                        opacity=0.7,
                        nbinsx=20
                    ))
                
                fig.update_layout(
                    title="測試集預測概率分布",
                    xaxis_title="預測概率",
                    yaxis_title="頻率",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**隨機森林配置：** {n_estimators}棵樹, 最大深度={max_depth}")

elif page == "📏 評價指標詳解":
    st.markdown('<h1 class="main-header">📏 評價指標詳解</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🎯 分類評價指標概述")
    st.info("💡 評價指標幫助我們量化分類模型的性能，選擇合適的指標對模型評估至關重要。")
    
    # 創建標籤頁式佈局
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔵 混淆矩陣", "🟡 準確率 & 精確率", "🟢 召回率 & F1分數", "🔴 ROC-AUC", "🟣 多分類指標"])
    
    with tab1:
        st.markdown("### 🔵 混淆矩陣 (Confusion Matrix)")
        
        st.markdown("混淆矩陣是分類問題評估的基礎，展示預測結果與真實標籤的對比：")
        
        # 二分類混淆矩陣
        st.markdown("#### 📊 二分類混淆矩陣")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 創建示例混淆矩陣
            cm_example = np.array([[85, 15], [10, 90]])
            fig = px.imshow(cm_example, 
                           text_auto=True,
                           x=['預測: 負類', '預測: 正類'],
                           y=['實際: 負類', '實際: 正類'],
                           title="二分類混淆矩陣示例",
                           color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**混淆矩陣術語：**")
            st.markdown("- **TN (True Negative)**: 85")
            st.markdown("- **FP (False Positive)**: 15") 
            st.markdown("- **FN (False Negative)**: 10")
            st.markdown("- **TP (True Positive)**: 90")
            
            st.info("**總準確率**: (85+90)/(85+15+10+90) = 87.5%")
        
        # 數學定義
        st.markdown("#### 📐 數學定義")
        st.latex(r'''
        \begin{pmatrix}
        TN & FP \\
        FN & TP
        \end{pmatrix}
        ''')
        
        st.markdown("其中：")
        st.markdown("- **TN**: 真陰性 - 正確預測為負類")
        st.markdown("- **FP**: 假陽性 - 錯誤預測為正類（第一類錯誤）")
        st.markdown("- **FN**: 假陰性 - 錯誤預測為負類（第二類錯誤）")
        st.markdown("- **TP**: 真陽性 - 正確預測為正類")
        
        # 混淆矩陣的指標選擇建議
        st.markdown("---")
        st.markdown("### 🎯 混淆矩陣的應用建議")
        st.success("""
        **何時使用混淆矩陣：**
        - 🔍 **詳細錯誤分析**：了解具體錯誤類型
        - 📊 **多分類問題**：查看類別間的混淆情況
        - 🎯 **模型診斷**：識別模型的弱點
        - 💡 **改進方向**：指導數據收集和特徵工程
        """)
    
    with tab2:
        st.markdown("### 🟡 準確率 (Accuracy) & 精確率 (Precision)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 準確率 (Accuracy)")
            st.markdown("準確率是最直觀的評價指標，表示預測正確的比例：")
            
            st.latex(r'''
            Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
            ''')
            
            st.success("### 📊 準確率特點")
            st.markdown("""
            - **範圍**: 0-1 (越高越好)
            - **適用**: 類別平衡的數據集
            - **優點**: 直觀易懂
            - **缺點**: 在不平衡數據中會誤導
            """)
        
        with col2:
            st.markdown("#### 🎯 精確率 (Precision)")
            st.markdown("精確率關注「預測為正類中，有多少真的是正類」：")
            
            st.latex(r'''
            Precision = \frac{TP}{TP + FP}
            ''')
            
            st.warning("### 📊 精確率特點")
            st.markdown("""
            - **範圍**: 0-1 (越高越好)
            - **關注**: 減少假陽性
            - **適用**: 假陽性代價高的場景
            - **例子**: 垃圾郵件偵測
            """)
        
        # 實際計算示例
        st.markdown("#### 🧮 實際計算示例")
        
        X, y, target_names = get_current_data()
        
        if len(X) > 0 and len(target_names) >= 2:
            # 簡單訓練一個模型進行演示
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # 計算指標
            acc = accuracy_score(y_test, y_pred)
            
            if len(target_names) == 2:
                prec = precision_score(y_test, y_pred)
            else:
                prec = precision_score(y_test, y_pred, average='weighted')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("準確率", f"{acc:.4f}")
            with col2:
                st.metric("精確率", f"{prec:.4f}")
        
        # 準確率和精確率的指標選擇建議
        st.markdown("---")
        st.markdown("### 🎯 準確率 & 精確率的選擇建議")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **選擇準確率當：**
            - ✅ 類別相對平衡
            - ✅ 所有錯誤代價相同
            - ✅ 需要整體性能概覽
            - ✅ 向利益相關者報告
            """)
        
        with col2:
            st.warning("""
            **選擇精確率當：**
            - ⚠️ 假陽性代價很高
            - 📧 垃圾郵件偵測
            - 🛒 推薦系統
            - 🎯 需要高確信度的預測
            """)
    
    with tab3:
        st.markdown("### 🟢 召回率 (Recall) & F1分數")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 召回率 (Recall)")
            st.markdown("召回率關注「實際正類中，有多少被正確預測」：")
            
            st.latex(r'''
            Recall = \frac{TP}{TP + FN}
            ''')
            
            st.success("### 📊 召回率特點")
            st.markdown("""
            - **範圍**: 0-1 (越高越好)
            - **關注**: 減少假陰性
            - **適用**: 假陰性代價高的場景
            - **例子**: 疾病診斷
            """)
        
        with col2:
            st.markdown("#### ⚖️ F1分數")
            st.markdown("F1分數是精確率和召回率的調和平均：")
            
            st.latex(r'''
            F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
            ''')
            
            st.info("### 📊 F1分數特點")
            st.markdown("""
            - **範圍**: 0-1 (越高越好)
            - **平衡**: 精確率和召回率
            - **適用**: 不平衡數據集
            - **優點**: 綜合考慮兩個指標
            """)
        
        # Precision-Recall權衡
        st.markdown("#### ⚖️ Precision-Recall 權衡")
        st.markdown("精確率和召回率通常存在權衡關係：")
        
        # 創建示例權衡圖
        thresholds = np.linspace(0.1, 0.9, 20)
        precision_sim = 1 - 0.5 * thresholds + 0.1 * np.random.randn(20)
        recall_sim = thresholds + 0.1 * np.random.randn(20)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precision_sim, mode='lines+markers', name='Precision'))
        fig.add_trace(go.Scatter(x=thresholds, y=recall_sim, mode='lines+markers', name='Recall'))
        fig.update_layout(
            title="Precision vs Recall 權衡",
            xaxis_title="分類閾值",
            yaxis_title="分數",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 召回率和F1分數的指標選擇建議
        st.markdown("---")
        st.markdown("### 🎯 召回率 & F1分數的選擇建議")
        
        col1, col2 = st.columns(2)
        with col1:
            st.error("""
            **選擇召回率當：**
            - 🚨 假陰性代價極高
            - 🩺 疾病診斷
            - 🔍 欺詐檢測
            - 🛡️ 安全系統
            """)
        
        with col2:
            st.info("""
            **選擇F1分數當：**
            - ⚖️ 需要平衡精確率和召回率
            - 📊 數據不平衡
            - 🎯 綜合評估模型
            - 🏆 模型比較
            """)
    
    with tab4:
        st.markdown("### 🔴 ROC曲線與AUC")
        
        st.markdown("#### 📈 ROC曲線 (Receiver Operating Characteristic)")
        st.markdown("ROC曲線展示不同閾值下TPR與FPR的關係：")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.latex(r'''
            \begin{align}
            TPR &= \frac{TP}{TP + FN} = Recall \\
            FPR &= \frac{FP}{FP + TN}
            \end{align}
            ''')
            
            st.markdown("**TPR (真陽性率)**: 召回率")
            st.markdown("**FPR (假陽性率)**: 假陽性佔所有陰性的比例")
        
        with col2:
            st.markdown("#### 📊 AUC (Area Under Curve)")
            st.latex(r'''
            AUC = \int_0^1 TPR(FPR^{-1}(x)) dx
            ''')
            
            st.markdown("**AUC範圍**:")
            st.markdown("- **AUC = 1**: 完美分類器")
            st.markdown("- **AUC = 0.5**: 隨機分類器")
            st.markdown("- **AUC < 0.5**: 比隨機還差")
        
        # 創建示例ROC曲線
        fpr_sim = np.linspace(0, 1, 100)
        tpr_sim = np.sqrt(fpr_sim) + 0.2 * np.random.randn(100)
        tpr_sim = np.clip(tpr_sim, 0, 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_sim, y=tpr_sim, mode='lines', name='ROC曲線'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='隨機分類器'))
        fig.update_layout(
            title="ROC曲線示例",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **ROC-AUC適用於**: 二分類問題，類別相對平衡的情況")
        
        # ROC-AUC的指標選擇建議
        st.markdown("---")
        st.markdown("### 🎯 ROC-AUC的選擇建議")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **選擇ROC-AUC當：**
            - 🎯 二分類問題
            - ⚖️ 類別相對平衡
            - 📈 關注整體排序能力
            - 🔄 需要調整分類閾值
            """)
        
        with col2:
            st.warning("""
            **避免ROC-AUC當：**
            - ⚠️ 嚴重類別不平衡
            - 📊 多分類問題
            - 🎯 關注特定類別性能
            - 💡 推薦使用PR-AUC替代
            """)
    
    with tab5:
        st.markdown("### 🟣 多分類問題指標")
        
        st.markdown("多分類問題需要特殊的評價方式：")
        
        # 宏平均 vs 微平均
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 宏平均 (Macro Average)")
            st.latex(r'''
            Macro = \frac{1}{n} \sum_{i=1}^{n} Metric_i
            ''')
            
            st.markdown("**特點：**")
            st.markdown("- 每個類別權重相等")
            st.markdown("- 受少數類別影響大")
            st.markdown("- 適合類別平衡數據")
        
        with col2:
            st.markdown("#### 📊 微平均 (Micro Average)")
            st.latex(r'''
            Micro = \frac{\sum_{i=1}^{n} TP_i}{\sum_{i=1}^{n} (TP_i + FP_i)}
            ''')
            
            st.markdown("**特點：**")
            st.markdown("- 樣本權重相等")
            st.markdown("- 受多數類別影響大")
            st.markdown("- 適合不平衡數據")
        
        # 多分類混淆矩陣示例
        st.markdown("#### 📊 多分類混淆矩陣")
        
        if len(target_names) > 2:
            # 使用真實數據創建混淆矩陣
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(cm, 
                           text_auto=True,
                           x=target_names,
                           y=target_names,
                           title=f"多分類混淆矩陣 - {dataset_choice}")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 計算各種平均指標
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Macro F1", f"{macro_f1:.4f}")
            with col2:
                st.metric("Micro F1", f"{micro_f1:.4f}")
            with col3:
                st.metric("Weighted F1", f"{weighted_f1:.4f}")
        
        # 多分類指標的選擇建議
        st.markdown("---")
        st.markdown("### 🎯 多分類指標的選擇建議")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **選擇Macro平均當：**
            - ⚖️ 每個類別同等重要
            - 🔍 想了解少數類別表現
            - 📊 類別相對平衡
            - 🎯 關注模型對所有類別的能力
            """)
        
        with col2:
            st.info("""
            **選擇Micro/Weighted平均當：**
            - 📈 關注整體準確性
            - 🎯 樣本數量重要
            - ⚠️ 類別不平衡
            - 💼 業務關注多數類別
            """)
    
    # 移動到每個標籤頁內的指標選擇建議已在上面實現

elif page == "🚀 梯度提升分類":
    st.markdown('<h1 class="main-header">🚀 梯度提升分類 (Gradient Boosting Classification)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🚀 梯度提升的基本概念")
    st.markdown("梯度提升通過逐步添加弱學習器來減少預測錯誤：")
    
    st.latex(r'''
    F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)
    ''')
    
    st.markdown("其中：")
    st.markdown("- $F_m(x)$: 第m步的強學習器")
    st.markdown("- $h_m(x)$: 第m個弱學習器")
    st.markdown("- γₘ: 學習率")
    
    st.markdown("### 📊 損失函數")
    st.latex(r'''
    L(y, F(x)) = -\sum_{k=1}^{K} y_k \log(p_k(x))
    ''')
    
    # 優缺點
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🎯 **預測準確率高**：通常表現優異
        - 📊 **處理混合數據**：數值+類別特徵
        - 🛡️ **對離群值穩健**：基於殘差學習
        - 🔧 **自動特徵選擇**：重要特徵優先
        - 📈 **提供特徵重要性**：可解釋性
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⏱️ **訓練時間長**：順序學習
        - ⚠️ **容易過擬合**：需要調參
        - 🎛️ **超參數多**：調參複雜
        - 💿 **記憶體需求大**：存儲多個模型
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("弱學習器數量：", 10, 200, 100, 10)
            learning_rate = st.slider("學習率：", 0.01, 0.3, 0.1, 0.01)
        
        with col2:
            max_depth = st.slider("樹的最大深度：", 1, 10, 3)
            test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:6]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42
            )
            
            # 建立模型
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            
            with st.spinner('訓練梯度提升模型中...'):
                model.fit(X_train, y_train)
            
            # 預測
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # 評估指標
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            if len(target_names) == 2:
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
            else:
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # 顯示結果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train-Data 準確率", f"{train_acc:.4f}")
            with col2:
                st.metric("Test-Data 準確率", f"{test_acc:.4f}")
            with col3:
                st.metric("Train-Data F1", f"{train_f1:.4f}")
            with col4:
                st.metric("Test-Data F1", f"{test_f1:.4f}")
            
            # 混淆矩陣
            st.markdown("### 📊 混淆矩陣")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_train = confusion_matrix(y_train, y_train_pred)
                fig = px.imshow(cm_train, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Training Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig = px.imshow(cm_test, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Test Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # 特徵重要性分析
            st.markdown("### 📊 特徵重要性分析")
            
            feature_importance = pd.DataFrame({
                '特徵': selected_features,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(x=feature_importance['特徵'], y=feature_importance['重要性'],
                       marker_color='orange')
            ])
            fig.update_layout(
                title=f"梯度提升特徵重要性 (學習率={learning_rate})",
                xaxis_title="特徵",
                yaxis_title="重要性",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 學習曲線
            st.markdown("### 📈 學習曲線")
            
            train_scores = []
            test_scores = []
            
            for i, (train_pred, test_pred) in enumerate(zip(
                model.staged_predict(X_train), 
                model.staged_predict(X_test)
            )):
                train_scores.append(accuracy_score(y_train, train_pred))
                test_scores.append(accuracy_score(y_test, test_pred))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(train_scores)+1)), y=train_scores, 
                                   mode='lines', name='Training Accuracy'))
            fig.add_trace(go.Scatter(x=list(range(1, len(test_scores)+1)), y=test_scores, 
                                   mode='lines', name='Test Accuracy'))
            fig.update_layout(
                title="梯度提升學習曲線",
                xaxis_title="迭代次數",
                yaxis_title="準確率",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 支持向量機":
    st.markdown('<h1 class="main-header">🎯 支持向量機 (Support Vector Machine)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🎯 SVM的基本概念")
    st.markdown("SVM尋找最大間隔的決策邊界：")
    
    st.latex(r'''
    \max \frac{2}{||\mathbf{w}||} \quad \text{subject to } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1
    ''')
    
    st.markdown("### 📊 核函數")
    st.markdown("常用核函數：")
    
    st.latex(r'''
    \begin{align}
    \text{線性核：} K(x, x') &= x \cdot x' \\
    \text{多項式核：} K(x, x') &= (x \cdot x' + c)^d \\
    \text{RBF核：} K(x, x') &= \exp(-\gamma ||x - x'||^2)
    \end{align}
    ''')
    
    # 優缺點
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🎯 **高維空間有效**：適合特徵多的數據
        - 🛡️ **記憶體效率高**：只使用支持向量
        - 🔧 **核技巧**：處理非線性問題
        - 📊 **泛化能力強**：最大間隔原理
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⏱️ **大數據集慢**：時間複雜度高
        - 🎛️ **參數敏感**：需要調整C和gamma
        - 📈 **不提供概率**：需要額外計算
        - 🔊 **對噪聲敏感**：離群值影響大
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            kernel = st.selectbox("核函數：", ["linear", "poly", "rbf", "sigmoid"])
            C = st.slider("正則化參數 C：", 0.01, 10.0, 1.0, 0.01)
        
        with col2:
            if kernel in ["poly", "rbf", "sigmoid"]:
                gamma = st.selectbox("Gamma：", ["scale", "auto"])
            else:
                gamma = "scale"
            test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:4]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42
            )
            
            # 標準化（SVM需要標準化）
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 建立模型
            model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                random_state=42
            )
            
            with st.spinner('訓練SVM模型中...'):
                model.fit(X_train_scaled, y_train)
            
            # 預測
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # 評估指標
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            if len(target_names) == 2:
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
            else:
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # 顯示結果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train-Data 準確率", f"{train_acc:.4f}")
            with col2:
                st.metric("Test-Data 準確率", f"{test_acc:.4f}")
            with col3:
                st.metric("Train-Data F1", f"{train_f1:.4f}")
            with col4:
                st.metric("Test-Data F1", f"{test_f1:.4f}")
            
            # 支持向量信息
            st.info(f"**支持向量數量**: {len(model.support_)} / {len(X_train)} ({len(model.support_)/len(X_train)*100:.1f}%)")
            
            # 混淆矩陣
            st.markdown("### 📊 混淆矩陣")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_train = confusion_matrix(y_train, y_train_pred)
                fig = px.imshow(cm_train, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Training Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig = px.imshow(cm_test, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Test Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

elif page == "🧮 貝葉斯分類器":
    st.markdown('<h1 class="main-header">🧮 貝葉斯分類器 (Naive Bayes Classifier)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 📊 貝葉斯定理")
    st.latex(r'''
    P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}
    ''')
    
    st.markdown("### 🔧 朴素假設")
    st.markdown("假設特徵之間條件獨立：")
    
    st.latex(r'''
    P(X|y) = \prod_{i=1}^{n} P(x_i|y)
    ''')
    
    # 優缺點
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - ⚡ **訓練和預測快速**：線性時間複雜度
        - 📊 **小樣本表現好**：不需要大量數據
        - 🎯 **多分類天然支持**：直接計算概率
        - 🛡️ **對離群值不敏感**：基於概率
        - 💾 **記憶體需求小**：只需存儲統計量
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 🔗 **獨立性假設強**：特徵相關時性能差
        - 📈 **線性決策邊界**：無法處理複雜關係
        - 🎲 **概率估計偏差**：可能過於自信
        - 📊 **連續特徵需要假設分布**：通常假設高斯
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**高斯朴素貝葉斯參數：**")
            var_smoothing = st.slider("方差平滑參數：", 1e-12, 1e-6, 1e-9, 1e-12)
        
        with col2:
            test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
            random_seed = st.slider("隨機種子：", 1, 100, 42)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:4]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=random_seed
            )
            
            # 建立模型
            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(X_train, y_train)
            
            # 預測
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_proba = model.predict_proba(X_train)
            y_test_proba = model.predict_proba(X_test)
            
            # 評估指標
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            if len(target_names) == 2:
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
            else:
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # 顯示結果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train-Data 準確率", f"{train_acc:.4f}")
            with col2:
                st.metric("Test-Data 準確率", f"{test_acc:.4f}")
            with col3:
                st.metric("Train-Data F1", f"{train_f1:.4f}")
            with col4:
                st.metric("Test-Data F1", f"{test_f1:.4f}")
            
            # 混淆矩陣
            st.markdown("### 📊 混淆矩陣")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_train = confusion_matrix(y_train, y_train_pred)
                fig = px.imshow(cm_train, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Training Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig = px.imshow(cm_test, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Test Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # 預測概率分析
            st.markdown("### 📈 預測概率分析")
            
            # 計算平均預測概率
            avg_proba = np.mean(y_test_proba, axis=0)
            
            fig = go.Figure(data=[
                go.Bar(x=target_names, y=avg_proba, marker_color='lightblue')
            ])
            fig.update_layout(
                title="各類別平均預測概率",
                xaxis_title="類別",
                yaxis_title="平均概率",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🧠 神經網路分類":
    st.markdown('<h1 class="main-header">🧠 神經網路分類 (Neural Network Classification)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🧠 多層感知器 (MLP)")
    st.markdown("神經網路通過多層非線性變換學習複雜模式：")
    
    st.latex(r'''
    \begin{align}
    z^{(l)} &= W^{(l)} a^{(l-1)} + b^{(l)} \\
    a^{(l)} &= \sigma(z^{(l)})
    \end{align}
    ''')
    
    st.markdown("### 📊 激活函數")
    st.latex(r'''
    \begin{align}
    \text{ReLU：} \sigma(x) &= \max(0, x) \\
    \text{Sigmoid：} \sigma(x) &= \frac{1}{1 + e^{-x}} \\
    \text{Tanh：} \sigma(x) &= \frac{e^x - e^{-x}}{e^x + e^{-x}}
    \end{align}
    ''')
    
    # 優缺點
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🌊 **學習非線性關係**：複雜決策邊界
        - 🎯 **通用近似器**：理論上可擬合任何函數
        - 🔧 **自動特徵學習**：隱藏層提取特徵
        - 📊 **適應性強**：適用各種問題
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⏱️ **訓練時間長**：迭代優化過程
        - 🎛️ **超參數多**：需要仔細調參
        - 🔍 **黑盒模型**：難以解釋
        - 📊 **需要大量數據**：避免過擬合
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            hidden_layer_sizes = st.selectbox("隱藏層結構：", [
                (50,), (100,), (50, 50), (100, 50), (100, 100)
            ])
            activation = st.selectbox("激活函數：", ["relu", "tanh", "logistic"])
        
        with col2:
            learning_rate_init = st.slider("初始學習率：", 0.001, 0.1, 0.001, 0.001)
            max_iter = st.slider("最大迭代次數：", 100, 1000, 200, 100)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:6]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
            
            # 標準化（神經網路必須標準化）
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 建立模型
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=42
            )
            
            with st.spinner('訓練神經網路模型中...'):
                model.fit(X_train_scaled, y_train)
            
            # 預測
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # 評估指標
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            if len(target_names) == 2:
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
            else:
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # 顯示結果
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Train-Data 準確率", f"{train_acc:.4f}")
            with col2:
                st.metric("Test-Data 準確率", f"{test_acc:.4f}")
            with col3:
                st.metric("Train-Data F1", f"{train_f1:.4f}")
            with col4:
                st.metric("Test-Data F1", f"{test_f1:.4f}")
            
            # 模型收斂信息
            if hasattr(model, 'n_iter_'):
                st.info(f"**收斂信息**: 迭代{model.n_iter_}次後收斂")
            
            # 混淆矩陣
            st.markdown("### 📊 混淆矩陣")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_train = confusion_matrix(y_train, y_train_pred)
                fig = px.imshow(cm_train, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Training Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cm_test = confusion_matrix(y_test, y_test_pred)
                fig = px.imshow(cm_test, 
                               text_auto=True, 
                               x=target_names, 
                               y=target_names,
                               title="Test Data 混淆矩陣")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # 損失曲線
            if hasattr(model, 'loss_curve_'):
                st.markdown("### 📈 損失曲線")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(model.loss_curve_)+1)), 
                    y=model.loss_curve_,
                    mode='lines',
                    name='Training Loss'
                ))
                fig.update_layout(
                    title="神經網路訓練損失曲線",
                    xaxis_title="迭代次數",
                    yaxis_title="損失值",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔄 交叉驗證與穩定性":
    st.markdown('<h1 class="main-header">🔄 交叉驗證與穩定性分析</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🔄 交叉驗證原理")
    
    st.markdown("### 📊 K折交叉驗證")
    st.markdown("將數據分成K個子集，輪流作為測試集：")
    
    st.latex(r'''
    CV_{score} = \frac{1}{K} \sum_{i=1}^{K} Score_i
    ''')
    
    st.markdown("### 🎯 交叉驗證的優點")
    st.info("""
    - 🎯 **更可靠的評估**：使用所有數據進行驗證
    - 📊 **穩定性測試**：觀察性能變化
    - 🔧 **減少過擬合風險**：多次驗證
    - 📈 **置信區間**：提供性能範圍
    """)
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            cv_folds = st.slider("交叉驗證折數：", 3, 10, 5)
            model_choice = st.selectbox("選擇模型：", [
                "邏輯回歸", "隨機森林", "支持向量機", "K近鄰"
            ])
        
        with col2:
            test_size = st.slider("最終測試集比例：", 0.1, 0.3, 0.2, 0.05)
            random_seed = st.slider("隨機種子：", 1, 100, 42)
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:6]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=random_seed
            )
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 選擇模型
            if model_choice == "邏輯回歸":
                model = LogisticRegression(random_state=random_seed, max_iter=1000)
            elif model_choice == "隨機森林":
                model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
            elif model_choice == "支持向量機":
                model = SVC(random_state=random_seed)
            else:  # K近鄰
                model = KNeighborsClassifier(n_neighbors=5)
            
            # 交叉驗證
            with st.spinner('執行交叉驗證中...'):
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=cv_folds, scoring='accuracy'
                )
            
            # 最終模型訓練
            model.fit(X_train_scaled, y_train)
            final_score = model.score(X_test_scaled, y_test)
            
            # 結果顯示
            st.markdown("### 📊 交叉驗證結果")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("CV平均分數", f"{cv_scores.mean():.4f}")
            with col2:
                st.metric("CV標準差", f"{cv_scores.std():.4f}")
            with col3:
                st.metric("最終測試分數", f"{final_score:.4f}")
            with col4:
                st.metric("穩定性評級", 
                         "優秀" if cv_scores.std() < 0.02 else 
                         "良好" if cv_scores.std() < 0.05 else "一般")
            
            # 交叉驗證分數分布
            st.markdown("### 📈 交叉驗證分數分布")
            
            fig = go.Figure()
            
            # 箱線圖
            fig.add_trace(go.Box(
                y=cv_scores,
                name='CV Scores',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            # 添加最終測試分數線
            fig.add_hline(y=final_score, line_dash="dash", line_color="red",
                         annotation_text=f"最終測試分數: {final_score:.4f}")
            
            fig.update_layout(
                title=f"{model_choice} - {cv_folds}折交叉驗證分數分布",
                yaxis_title="準確率",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 穩定性分析
            st.markdown("### 🔍 穩定性分析")
            
            stability_score = 1 - (cv_scores.std() / cv_scores.mean())
            
            if stability_score > 0.95:
                st.success(f"🎯 **模型穩定性優秀** (穩定性指標: {stability_score:.3f})")
            elif stability_score > 0.90:
                st.info(f"✅ **模型穩定性良好** (穩定性指標: {stability_score:.3f})")
            else:
                st.warning(f"⚠️ **模型穩定性一般** (穩定性指標: {stability_score:.3f})")

elif page == "⚖️ 資料不平衡處理":
    st.markdown('<h1 class="main-header">⚖️ 資料不平衡處理</h1>', unsafe_allow_html=True)
    
    st.markdown("## ⚖️ 資料不平衡問題")
    
    st.markdown("### 🎯 什麼是資料不平衡")
    st.markdown("當不同類別的樣本數量差異很大時，會導致模型偏向多數類別。")
    
    st.markdown("### 📊 常見處理方法")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 數據層面")
        st.markdown("""
        - **過採樣 (Over-sampling)**：增加少數類別樣本
        - **欠採樣 (Under-sampling)**：減少多數類別樣本  
        - **SMOTE**：合成少數類別樣本
        - **組合採樣**：結合過採樣和欠採樣
        """)
    
    with col2:
        st.markdown("#### 🎛️ 算法層面")
        st.markdown("""
        - **類別權重**：給少數類別更高權重
        - **集成方法**：Balanced Random Forest
        - **閾值調整**：優化分類閾值
        - **代價敏感學習**：不同錯誤不同代價
        """)
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 檢查當前數據集的平衡性
        class_counts = pd.Series(y).value_counts().sort_index()
        imbalance_ratio = class_counts.min() / class_counts.max()
        
        st.markdown("### 📊 當前數據集平衡性分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("不平衡比例", f"{imbalance_ratio:.3f}")
            
            if imbalance_ratio < 0.1:
                st.error("嚴重不平衡")
            elif imbalance_ratio < 0.5:
                st.warning("中度不平衡")
            else:
                st.success("相對平衡")
        
        with col2:
            # 類別分布圖
            fig = go.Figure(data=[
                go.Bar(x=target_names, y=class_counts.values)
            ])
            fig.update_layout(
                title="類別分布",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 處理方法選擇
        st.markdown("### 🔧 不平衡處理方法比較")
        
        processing_method = st.selectbox("選擇處理方法：", [
            "無處理", "SMOTE過採樣", "類別權重平衡"
        ])
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:4]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 應用處理方法
            if processing_method == "SMOTE過採樣":
                try:
                    smote = SMOTE(random_state=42)
                    X_train_processed, y_train_processed = smote.fit_resample(X_train_scaled, y_train)
                    st.success("✅ SMOTE過採樣完成")
                except:
                    X_train_processed, y_train_processed = X_train_scaled, y_train
                    st.error("❌ SMOTE失敗，使用原始數據")
            else:
                X_train_processed, y_train_processed = X_train_scaled, y_train
            
            # 建立模型
            if processing_method == "類別權重平衡":
                model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
            else:
                model = LogisticRegression(random_state=42, max_iter=1000)
            
            # 訓練和預測
            model.fit(X_train_processed, y_train_processed)
            y_pred = model.predict(X_test_scaled)
            
            # 評估
            accuracy = accuracy_score(y_test, y_pred)
            
            if len(target_names) == 2:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
            else:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            
            # 結果顯示
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("準確率", f"{accuracy:.4f}")
            with col2:
                st.metric("精確率", f"{precision:.4f}")
            with col3:
                st.metric("召回率", f"{recall:.4f}")
            with col4:
                st.metric("F1分數", f"{f1:.4f}")
            
            # 混淆矩陣
            st.markdown("### 📊 處理後的混淆矩陣")
            
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, 
                           text_auto=True, 
                           x=target_names, 
                           y=target_names,
                           title=f"混淆矩陣 - {processing_method}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 模型可解釋性":
    st.markdown('<h1 class="main-header">🔍 模型可解釋性分析</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🔍 為什麼需要可解釋性")
    
    st.info("""
    **模型可解釋性的重要性：**
    - 🏥 **醫療診斷**：醫生需要了解診斷依據
    - 🏦 **金融風控**：監管要求解釋決策過程
    - 🎯 **業務決策**：管理層需要理解模型邏輯
    - 🔧 **模型改進**：找出模型的不足之處
    """)
    
    st.markdown("### 📊 可解釋性方法分類")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 內在可解釋性")
        st.markdown("""
        - **線性模型**：係數直接反映特徵重要性
        - **決策樹**：規則路徑清晰可見
        - **貝葉斯模型**：概率推理過程透明
        """)
    
    with col2:
        st.markdown("#### 🔧 後處理可解釋性")
        st.markdown("""
        - **特徵重要性**：排序特徵貢獻度
        - **LIME**：局部線性近似
        - **SHAP**：博弈論解釋
        """)
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 模型選擇
        interpretable_model = st.selectbox("選擇可解釋模型：", [
            "邏輯回歸", "決策樹", "隨機森林"
        ])
        
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:6]
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 建立模型
            if interpretable_model == "邏輯回歸":
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train_scaled, y_train)
            elif interpretable_model == "決策樹":
                model = DecisionTreeClassifier(max_depth=5, random_state=42)
                model.fit(X_train, y_train)  # 決策樹不需要標準化
            else:  # 隨機森林
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)  # 隨機森林不需要標準化
            
            # 預測
            if interpretable_model == "邏輯回歸":
                y_pred = model.predict(X_test_scaled)
                accuracy = model.score(X_test_scaled, y_test)
            else:
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
            
            st.metric("模型準確率", f"{accuracy:.4f}")
            
            # 可解釋性分析
            st.markdown("### 🔍 模型可解釋性分析")
            
            if interpretable_model == "邏輯回歸":
                # 邏輯回歸係數分析
                if len(target_names) == 2:  # 二分類
                    coef_df = pd.DataFrame({
                        '特徵': selected_features,
                        '係數': model.coef_[0],
                        '絕對值': np.abs(model.coef_[0])
                    }).sort_values('絕對值', ascending=False)
                    
                    fig = go.Figure(data=[
                        go.Bar(x=coef_df['特徵'], y=coef_df['係數'],
                               marker_color=['red' if x < 0 else 'blue' for x in coef_df['係數']])
                    ])
                    fig.update_layout(
                        title="邏輯回歸特徵係數 (正數增加正類概率，負數減少)",
                        xaxis_title="特徵",
                        yaxis_title="係數值",
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 係數解釋
                    st.markdown("#### 📊 係數解釋")
                    for _, row in coef_df.head(3).iterrows():
                        if row['係數'] > 0:
                            st.success(f"**{row['特徵']}**: 係數 {row['係數']:.3f} - 增加此特徵會提高正類概率")
                        else:
                            st.error(f"**{row['特徵']}**: 係數 {row['係數']:.3f} - 增加此特徵會降低正類概率")
            
            elif interpretable_model in ["決策樹", "隨機森林"]:
                # 特徵重要性分析
                feature_importance = pd.DataFrame({
                    '特徵': selected_features,
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=False)
                
                fig = go.Figure(data=[
                    go.Bar(x=feature_importance['特徵'], y=feature_importance['重要性'],
                           marker_color='green')
                ])
                fig.update_layout(
                    title=f"{interpretable_model}特徵重要性",
                    xaxis_title="特徵",
                    yaxis_title="重要性",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 重要性解釋
                st.markdown("#### 📊 特徵重要性解釋")
                for _, row in feature_importance.head(3).iterrows():
                    st.info(f"**{row['特徵']}**: 重要性 {row['重要性']:.3f} - 對分類決策的貢獻度")
            
            # 預測示例解釋
            st.markdown("### 🎯 單個預測示例解釋")
            
            sample_idx = st.slider("選擇測試樣本：", 0, len(X_test)-1, 0)
            
            if interpretable_model == "邏輯回歸":
                sample_features = X_test_scaled[sample_idx]
                sample_pred_proba = model.predict_proba([sample_features])[0]
            else:
                sample_features = X_test.iloc[sample_idx].values
                sample_pred_proba = model.predict_proba([sample_features])[0]
            
            sample_pred = np.argmax(sample_pred_proba)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 樣本特徵值")
                for i, feature in enumerate(selected_features):
                    st.metric(feature, f"{X_test.iloc[sample_idx][feature]:.3f}")
            
            with col2:
                st.markdown("#### 🎯 預測結果")
                st.metric("預測類別", target_names[sample_pred])
                st.metric("預測概率", f"{sample_pred_proba[sample_pred]:.3f}")
                
                # 概率分布
                fig = go.Figure(data=[
                    go.Bar(x=target_names, y=sample_pred_proba)
                ])
                fig.update_layout(
                    title="各類別預測概率",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "🏆 模型綜合比較":
    st.markdown('<h1 class="main-header">🏆 模型綜合比較</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🏆 多模型性能比較")
    
    st.info("💡 本頁面將訓練多個分類模型並進行全面比較，幫助您選擇最適合的模型。")
    
    X, y, target_names = get_current_data()
    
    if len(X) > 0:
        # 參數設置
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("測試集比例：", 0.1, 0.4, 0.2, 0.05)
            cv_folds = st.slider("交叉驗證折數：", 3, 10, 5)
        
        with col2:
            random_seed = st.slider("隨機種子：", 1, 100, 42)
            
        # 特徵選擇
        selected_features = st.multiselect(
            "選擇用於建模的特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()
        )
        
        if len(selected_features) > 0:
            X_selected = X[selected_features]
            
            # 數據分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=random_seed
            )
            
            # 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 定義模型
            models = {
                '邏輯回歸': LogisticRegression(random_state=random_seed, max_iter=1000),
                'K近鄰': KNeighborsClassifier(n_neighbors=5),
                '決策樹': DecisionTreeClassifier(random_state=random_seed, max_depth=10),
                '隨機森林': RandomForestClassifier(n_estimators=100, random_state=random_seed),
                '梯度提升': GradientBoostingClassifier(n_estimators=100, random_state=random_seed),
                '支持向量機': SVC(random_state=random_seed, probability=True),
                '貝葉斯分類器': GaussianNB(),
                '神經網路': MLPClassifier(hidden_layer_sizes=(100,), random_state=random_seed, max_iter=500)
            }
            
            # 訓練和評估
            results = {}
            
            with st.spinner('訓練多個模型中...'):
                for name, model in models.items():
                    # 選擇是否需要標準化
                    if name in ['邏輯回歸', 'K近鄰', '支持向量機', '貝葉斯分類器', '神經網路']:
                        X_train_use = X_train_scaled
                        X_test_use = X_test_scaled
                    else:  # 決策樹、隨機森林、梯度提升不需要標準化
                        X_train_use = X_train
                        X_test_use = X_test
                    
                    # 訓練模型
                    model.fit(X_train_use, y_train)
                    
                    # 預測
                    y_pred = model.predict(X_test_use)
                    
                    # 評估指標
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    if len(target_names) == 2:
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                    else:
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # 交叉驗證
                    cv_scores = cross_val_score(model, X_train_use, y_train, cv=cv_folds, scoring='accuracy')
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
            
            # 結果展示
            st.markdown("### 📊 模型性能比較表")
            
            results_df = pd.DataFrame(results).T
            results_df = results_df.round(4)
            
            # 添加排名
            results_df['F1排名'] = results_df['f1'].rank(ascending=False).astype(int)
            results_df['準確率排名'] = results_df['accuracy'].rank(ascending=False).astype(int)
            
            st.dataframe(results_df, use_container_width=True)
            
            # 性能比較圖
            st.markdown("### 📈 模型性能可視化比較")
            
            # 準確率比較
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='測試準確率',
                x=list(results.keys()),
                y=[results[name]['accuracy'] for name in results.keys()],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='CV平均準確率',
                x=list(results.keys()),
                y=[results[name]['cv_mean'] for name in results.keys()],
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="模型準確率比較",
                xaxis_title="模型",
                yaxis_title="準確率",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # F1分數比較
            fig = go.Figure(data=[
                go.Bar(x=list(results.keys()), 
                       y=[results[name]['f1'] for name in results.keys()],
                       marker_color='lightgreen')
            ])
            fig.update_layout(
                title="模型F1分數比較",
                xaxis_title="模型",
                yaxis_title="F1分數",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 模型推薦
            st.markdown("### 🎯 模型選擇建議")
            
            best_f1_model = max(results.keys(), key=lambda x: results[x]['f1'])
            best_acc_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            most_stable_model = min(results.keys(), key=lambda x: results[x]['cv_std'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"""
                **🏆 最佳F1分數**
                
                **{best_f1_model}**
                
                F1: {results[best_f1_model]['f1']:.4f}
                """)
            
            with col2:
                st.info(f"""
                **🎯 最佳準確率**
                
                **{best_acc_model}**
                
                準確率: {results[best_acc_model]['accuracy']:.4f}
                """)
            
            with col3:
                st.warning(f"""
                **⚖️ 最穩定模型**
                
                **{most_stable_model}**
                
                CV標準差: {results[most_stable_model]['cv_std']:.4f}
                """)
            
            # 選擇建議
            st.markdown("### 💡 選擇建議")
            
            if results[best_f1_model]['f1'] > 0.9:
                st.success("🎉 有模型表現優異！建議選擇F1分數最高的模型。")
            elif results[most_stable_model]['cv_std'] < 0.02:
                st.info("⚖️ 模型穩定性良好，建議選擇最穩定的模型。")
            else:
                st.warning("⚠️ 模型性能一般，建議嘗試特徵工程或參數調優。")

else:
    st.markdown(f"# {page}")
    st.info("此頁面正在開發中，敬請期待！")

# 應用結束 