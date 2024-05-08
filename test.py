import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
# Định nghĩa hàm để đọc file CSV và trả về DataFrame
def load_data(file):
    data = pd.read_csv(file)
    return data

# Định nghĩa các hàm xử lý dữ liệu
def process_data(data, selected_columns, process_method):
    if process_method == "Chuyển chuỗi thành số":
        # Chuyển từ kiểu chuỗi thành kiểu số thực
        data[selected_columns] = data[selected_columns].astype(float)
    if process_method == "Xóa giá trị null":
        data[selected_columns] = data[selected_columns].dropna(subset=selected_columns)
    if process_method == "Thay giá trị null bằng giá trị TB":
        data[selected_columns] = data[selected_columns].applymap(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
        data[selected_columns] = data[selected_columns].astype(float)
        # Xác định giá trị trung bình của từng cột
        means = data[selected_columns].mean()
        # Thay thế các giá trị null bằng giá trị trung bình của cột tương ứng
        data[selected_columns] = data[selected_columns].fillna(means)
    if process_method == "Thay giá trị null bằng giá trị trung vị":
        data[selected_columns] = data[selected_columns].applymap(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
        data[selected_columns] = data[selected_columns].astype(float)
        # Xác định giá trị trung vị của từng cột
        medians = data[selected_columns].median()
        # Thay thế các giá trị null bằng giá trị trung vị của cột tương ứng
        data[selected_columns] = data[selected_columns].fillna(medians)
    if process_method == "Thay giá trị null bằng giá trị xuất hiện nhiều nhất":
        # Xác định giá trị xuất hiện nhiều nhất của từng cột
        modes = data[selected_columns].mode().iloc[0]
        # Thay thế các giá trị null bằng giá trị xuất hiện nhiều nhất của cột tương ứng
        data[selected_columns] = data[selected_columns].fillna(modes)
    if process_method == "Xóa ký tự đặc biệt":
        data[selected_columns] = data[selected_columns].applymap(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))
    if process_method == "Chuyển đổi %":
        data[selected_columns] /= 100

    return data

# Layout chính
def main_layout(data):
    st.title("CSV Data Processing Tool")

    # Hiển thị bảng dữ liệu gốc
    st.header("Original Data")
    st.dataframe(data)

# Layout cho phần xử lý dữ liệu
def data_processing_layout(data):
    st.header("Data Processing")
    # Chọn cột để xử lý
    selected_columns = st.multiselect("Select columns to process", data.columns)

    # Chọn cách thức xử lý
    process_method = st.selectbox("Select processing method", ["Chuyển chuỗi thành số", "Xóa ký tự đặc biệt", "Chuyển đổi %","Xóa giá trị null",
                                                               "Thay giá trị null bằng giá trị TB","Thay giá trị null bằng giá trị trung vị",
                                                               "Thay giá trị null bằng giá trị xuất hiện nhiều nhất"])

    # Hiển thị các nút trên cùng một hàng
    col1, col2, col3 = st.columns(3)
    with col1:
        process_button = st.button("Processing Data")
    with col2:
        save_button = st.button("Lưu Data")
    with col3:
        back_button = st.button("Trở lại")

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = data

    if process_button:
        processed_data = process_data(st.session_state.processed_data, selected_columns, process_method)
        st.session_state.processed_data = processed_data
        modes = data[selected_columns].mode().iloc[0]
        mode_info = pd.DataFrame({'colum': selected_columns, 'Xuất hiện nhiều nhất': modes})
        #st.dataframe(mode_info)

    if save_button:
        st.session_state.processed_data = processed_data

    if back_button:
        st.session_state.processed_data = data

    st.dataframe(st.session_state.processed_data)
    # Hiển thị bảng data types nằm ngang
    st.write("Data types:")
    dtypes_df = st.session_state.processed_data.dtypes.reset_index()
    dtypes_df.columns = ['Column', 'Data Type']
    st.dataframe(dtypes_df.T)
    # Tạo bảng thống kê các cột chứa giá trị null
    null_counts = st.session_state.processed_data.isnull().sum()
    null_counts_df = pd.DataFrame(null_counts, columns=['Null Count'])
    st.write("Null Value Counts:")
    st.dataframe(null_counts_df.T)
    global data1
    data1 = st.session_state.processed_data
def data_info(data):
    st.header("Data Information")
    # Hiển thị thông tin cơ bản về dữ liệu như số hàng, số cột, kiểu dữ liệu
    st.write("Number of rows:", data.shape[0])
    st.write("Number of columns:", data.shape[1])
    st.write("Data types:", data.dtypes)

def number_features_layout(data):
    st.header("Number Features")

    # Chọn cột kiểu số thực
    number_columns = [col for col in data.columns if data[col].dtype == 'float64' or data[col].dtype == 'int64']
    selected_column = st.selectbox("Select a numeric column", number_columns)

    if selected_column:
        # Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(data.index, data[selected_column])
        ax.set_xlabel("Index")
        ax.set_ylabel(selected_column)
        ax.set_title(f"Bar plot of {selected_column}")
        # Loại bỏ các thành phần viền
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)

def categorical_features_layout(data):
    st.header("Categorical Features")

    # Chọn cột kiểu chuỗi
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    selected_column = st.selectbox("Select a categorical column", categorical_columns)

    if selected_column:
        # Đếm số lần xuất hiện của từng giá trị trong cột
        value_counts = data[selected_column].value_counts()

        # Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(value_counts.index, value_counts.values)
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Bar plot of {selected_column}")
        # Loại bỏ các thành phần viền
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)


def number_and_categorical_features_layout(data):
    st.header("Number and Categorical Features")

    # Chọn cột kiểu số
    number_columns = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
    selected_number_column = st.selectbox("Select a numeric column", number_columns)

    # Chọn cột kiểu chuỗi
    categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
    selected_categorical_column = st.selectbox("Select a categorical column", categorical_columns)

    # Chọn loại biểu đồ
    plot_type = st.selectbox("Select plot type", ["Bar plot", "Histogram"])

    show_plot_button = st.button("Show Plot")

    if selected_number_column and selected_categorical_column and plot_type and show_plot_button:
        # Vẽ biểu đồ tương ứng
        if plot_type == "Bar plot":
            fig, ax = plt.subplots(figsize=(10, 6))

            # Biểu đồ thanh cho cột kiểu số
            if selected_categorical_column:
                value_counts = data.groupby(selected_categorical_column)[selected_number_column].mean()
                ax.bar(value_counts.index, value_counts.values)
                ax.set_xlabel(selected_categorical_column)
                ax.set_ylabel(selected_number_column)
                ax.set_title(f"Bar plot of {selected_number_column} by {selected_categorical_column}")

            # Loại bỏ các thành phần viền
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)

        elif plot_type == "Histogram":
            fig, ax = plt.subplots(figsize=(10, 6))

            # Histogram cho cột kiểu số
            if selected_categorical_column:
                for category in data[selected_categorical_column].unique():
                    ax.hist(data[data[selected_categorical_column] == category][selected_number_column], bins=20,
                            alpha=0.5, label=category)
                ax.set_xlabel(selected_number_column)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Histogram of {selected_number_column} by {selected_categorical_column}")
                ax.legend()

            # Loại bỏ các thành phần viền
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)

# Layout cho phần Random Forex
def random_forex_layout(data):
    st.header("Random Forest for Regression")

    # Chọn cột đầu vào dự đoán
    input_columns = st.multiselect("Select columns for X", data.select_dtypes(include=np.number).columns)

    # Chọn cột đầu ra cần dự đoán
    target_column = st.selectbox("Select column for Y", data.select_dtypes(include=np.number).columns)

    # Hiển thị nút để thực hiện dự đoán
    if st.button("Predict"):
        # Tạo X và y từ dữ liệu
        X = data[input_columns]
        y = data[target_column]

        # Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tạo và huấn luyện mô hình Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Đánh giá mô hình trên tập kiểm tra
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        # Vẽ biểu đồ so sánh giữa giá trị dự đoán và giá trị thực tế trên tập kiểm tra
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

# Layout cho phần Linear Regression Model
def linear_regression_layout(data):
    st.header("Linear Regression Model")

    # Chọn cột đầu vào dự đoán
    input_columns = st.multiselect("Select columns for X", data.select_dtypes(include=np.number).columns)

    # Chọn cột đầu ra cần dự đoán
    target_column = st.selectbox("Select column for Y", data.select_dtypes(include=np.number).columns)

    # Hiển thị nút để thực hiện dự đoán
    if st.button("Predict"):
        # Tạo X và y từ dữ liệu
        X = data[input_columns]
        y = data[target_column]

        # Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tạo và huấn luyện mô hình Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Đánh giá mô hình trên tập kiểm tra
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        # Vẽ biểu đồ so sánh giữa giá trị dự đoán và giá trị thực tế trên tập kiểm tra
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

# Layout cho phần Decision Tree Model
def decision_tree_layout(data):
    st.header("Decision Tree Model")

    # Chọn cột đầu vào dự đoán
    input_columns = st.multiselect("Select columns for X", data.select_dtypes(include=np.number).columns)

    # Chọn cột đầu ra cần dự đoán
    target_column = st.selectbox("Select column for Y", data.select_dtypes(include=np.number).columns)

    # Hiển thị nút để thực hiện dự đoán
    if st.button("Predict"):
        # Tạo X và y từ dữ liệu
        X = data[input_columns]
        y = data[target_column]

        # Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tạo và huấn luyện mô hình Decision Tree
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Đánh giá mô hình trên tập kiểm tra
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        # Vẽ biểu đồ so sánh giữa giá trị dự đoán và giá trị thực tế trên tập kiểm tra
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        # Vẽ cây quyết định
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model, ax=ax, feature_names=input_columns, filled=True)
        st.pyplot(fig)

def standardized_data_layout(data):
    st.header("Standardized Data")

    # Chọn duy nhất một cột
    target_column = st.selectbox("Select target column", data.columns)

    # Tạo một DataFrame tạm thời để xác định các loại phần tử trong cột đã chọn
    temp_df = data[target_column].value_counts().reset_index()
    temp_df.columns = ['Value', 'Count']

    # Chọn các loại phần tử có trong cột
    selected_values = st.multiselect("Select values to standardize", temp_df['Value'].tolist())

    # Chọn các số tương ứng
    numbers = list(range(1, len(selected_values) + 1))
    target_values = st.multiselect("Select corresponding values", numbers, default=numbers)

    # Hiển thị nút để thực hiện chuyển đổi
    if st.button("Standardize Data"):
        # Tạo một bản sao của dữ liệu
        processed_data = data.copy()

        # Chuyển đổi các phần tử được chọn thành các số tương ứng
        mapping = dict(zip(selected_values, target_values))
        processed_data[target_column] = processed_data[target_column].replace(mapping)

        # Hiển thị dữ liệu đã chuyển đổi
        st.dataframe(processed_data)

def main():

    st.set_page_config(layout="wide")  # Thiết lập trang Streamlit để có giao diện rộng
    st.sidebar.title("CSV File Upload")

    # Cho phép người dùng tải lên file CSV
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        main_layout(data)

    sidebar_options1 = ["None","Data Info", "Data processing"]
    selected_option1 = st.sidebar.selectbox("Tiền xử lý", sidebar_options1)
    sidebar_options2 = ["None","Number Features", "Categorical Features", "Number and Categorical"]
    selected_option2 = st.sidebar.selectbox("Các biến", sidebar_options2)
    sidebar_options = ["None","Standardized data"]
    selected_option = st.sidebar.selectbox("Chuân hóa", sidebar_options)
    sidebar_options3 = ["None","Random Forex", "Linear Regression", "Descion Tree"]
    selected_option3 = st.sidebar.selectbox("Model", sidebar_options3)

    if selected_option1 == "Data processing":
        if uploaded_file is not None:
            data_processing_layout(data)

    if selected_option1 == "Data Info":
        if uploaded_file is not None:
            data_info(data)
    if selected_option2 == "Number Features":
            if uploaded_file is not None:
                number_features_layout(data1)
    if selected_option2 == "Categorical Features":
        if uploaded_file is not None:
            categorical_features_layout(data1)
    if selected_option2 == "Number and Categorical":
        if uploaded_file is not None:
            number_and_categorical_features_layout(data1)
    if selected_option == "Standardized data":
        if uploaded_file is not None:
            standardized_data_layout(data1)
    if selected_option3 == "Random Forex":
        if uploaded_file is not None:
            random_forex_layout(data1)
    if selected_option3 == "Linear Regression":
        if uploaded_file is not None:
            linear_regression_layout(data1)
    if selected_option3 == "Descion Tree":
        if uploaded_file is not None:
            decision_tree_layout(data1)

if __name__ == "__main__":
    main()
