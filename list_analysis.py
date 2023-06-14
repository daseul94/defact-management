import altair as alt
import pandas as pd
import streamlit as st


def listview():
    df = pd.read_csv("data/result.csv")
    df.sort_values(by="일자", inplace=True, ascending=False, ignore_index=True)
    df = df.fillna("-")
    st.dataframe(df, width=800, height=800)


def _preprocess(df_data: pd.DataFrame):
    def _trim_locate(text):
        text = text.strip()
        text = text.replace("주망", "주방")
        return text

    df_data["위치"] = df_data["위치"].apply(lambda x: _trim_locate(x))
    return df_data


def get_frequency_dataframe(df_data: pd.DataFrame, column: str):
    values = df_data[column].value_counts()
    values = {k: v for k, v in values.items() if k and k not in ["-"]}
    values = pd.DataFrame({
        a: v for a, v in zip([column, "발생 건수"], zip(*values.items()))}
    )
    values.sort_values(by="발생 건수", inplace=True, ascending=False, ignore_index=True)
    return values


def get_frequency_graph(df_data: pd.DataFrame, column: str, fill: str = "orange", width: int = 300,
                        label_name: str = "발생 건수"):
    values = get_frequency_dataframe(df_data, column)
    values.iloc[4:, 0] = "기타"
    return alt.Chart(values, width=width).mark_bar(fill=fill).encode(
        x=alt.X(column, sort=None),
        y=label_name,
    )


def analysis():
    df = pd.read_csv("data/result.csv")
    df = _preprocess(df)

    values = {}
    values["위치"] = get_frequency_dataframe(df, "위치")
    values["부위"] = get_frequency_dataframe(df, "부위")
    values["공종"] = get_frequency_dataframe(df, "공종")
    values["하자유형"] = get_frequency_dataframe(df, "하자유형")

    columns = st.columns(4)
    targets = ["위치", "부위", "공종", "하자유형"]
    colors = ["orange", "green", "blue", "red"]

    assert len(columns) == len(targets) == len(colors), \
        "columns, targets, colors must have same length"

    for col, target, color in zip(columns, targets, colors):
        col.altair_chart(
            get_frequency_graph(df, target, fill=color),
            use_container_width=True,
        )

def main():
    st.set_page_config(layout="wide")
    view_defect_list, view_analysis = st.tabs(
        ["접수 목록", "분석"]
    )

    with view_defect_list:
        listview()

    with view_analysis:
        analysis()


if __name__ == "__main__":
    main()
