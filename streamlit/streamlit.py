import streamlit as st
import numpy as np
from PIL import Image
import random
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import logging
from logging.handlers import RotatingFileHandler
import os
import requests

api_url = "http://127.0.0.1:8000/generate/"

# Настройка логирования
log_directory = "logs"  
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, "streamlit.log")

# Создание обработчика для ротации логов
handler = RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[handler]
)

logger = logging.getLogger()


if "eda_clicked" not in st.session_state:
    st.session_state.eda_clicked = False
if "dataset_for_EDA" not in st.session_state:
    st.session_state.dataset_for_EDA = None
if "info_clicked" not in st.session_state:
    st.session_state.info_clicked = False

themes = [
    "plotly",
    "ggplot2",
    "seaborn",
    "simple_white",
    "xgridoff",
    "presentation",
    "streamlit",
]


def dataset_to_eda(df):
    dataset_transform = pd.DataFrame(columns=[])
    dataset_transform["epitets_num"] = df["text"].apply(lambda x: len(x.split(",")[1:]))
    dataset_transform["description"] = df["text"].apply(
        lambda x: ",".join(x.split(",")[1:])
    )
    dataset_transform["len"] = df["text"].apply(lambda x: len(x))

    dataset_transform["shape"] = df["image"].apply(lambda x: np.array(x).shape)
    dataset_transform["h"] = df["image"].apply(lambda x: np.array(x).shape[0])
    dataset_transform["w"] = df["image"].apply(lambda x: np.array(x).shape[1])
    dataset_transform["rgb"] = df["image"].apply(
        lambda x: "RGB" if len(np.array(x).shape) == 3 else "BW"
    )
    dataset_transform["ratio"] = dataset_transform["shape"].apply(lambda x: x[0] / x[1])
    dataset_transform["pixel"] = dataset_transform["shape"].apply(lambda x: np.prod(x))
    return dataset_transform


st.title("Генерация логотипа по текстовому описанию")

description = st.text_input("Введите описание логотипа:")

if st.button("Сгенерировать логотип"):
    if description:
        generated_images = []
        logger.info("Начинается генерация логотипа с описанием: %s", description)
        for i in range(4):
            payload = {"description": description}
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                # Конвертация изображения из байтов
                response_list = response.json()
                pixel_data = np.array(response_list["image"], dtype=np.uint8)
                generate_image = Image.fromarray(pixel_data)
                generated_images.append(generate_image)
                logger.info("Логотип %d сгенерирован успешно", i + 1)
            else:
                logger.error(f"Ошибка при подключении к API: {response.status_code}")
                st.write("Ошибка при подключении к API")
        st.session_state.generate_image = generated_images
        st.session_state.description = description
        logger.info("Генерация логотипов завершена.")
    else:
        st.warning("Пожалуйста, введите описание.")
        logger.warning("Пользователь не ввел описание.")

if "generate_image" in st.session_state:
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.image(
            st.session_state.generate_image[0],
            caption="Сгенерированный логотип #1",
            width=124,
        )
        logger.info("Отображается %s", f"Сгенерированный логотип #{i + 1}")

    with col2:
        st.image(
            st.session_state.generate_image[1],
            caption="Сгенерированный логотип #2",
            width=124,
        )
    with col3:
        st.image(
            st.session_state.generate_image[2],
            caption="Сгенерированный логотип #3",
            width=124,
        )
    with col4:
        st.image(
            st.session_state.generate_image[3],
            caption="Сгенерированный логотип #4",
            width=124,
        )

with st.expander("Краткая информация о модели: свернуть/развернуть", expanded=True):
    if st.button("Краткая информация о модели"):
        st.session_state.info_clicked = True
        logger.info("Пользователь запросил краткую информацию о модели.")

    if st.session_state.info_clicked:
        st.header("Краткая справка")
        st.markdown(
            f"Модель GAN состоит из двух нейросетей: **генератора** и **дискриминатора**. "
            f"И генератор, и дискриминатор представляют собой свёрточные нейросети."
            f" Генератор обучается создавать изображения из шума и текстового описания, "
            f"а дискриминатор учится распознавать сгенерированные изображения."
            f" Обучение ввиду ограниченных ресурсов длилось всего 15 эпох."
            f" Для оценки качества генерации использовалась метрика FID."
            f" Ссылка на [репозиторий GitHub](https://github.com/HerrVonBeloff/AI-YP_24-team-42).",
            unsafe_allow_html=True,
        )
        logger.info("Показана краткая справка о модели.")
        st.header("Процесс обучения")
        theme_model = st.selectbox(
            "Выберите тему для графиков", sorted(themes), key="theme_selector1"
        )
        df_loss = pd.read_csv("loss_long.csv")
        df_metric = pd.read_csv("metric.csv")
        logger.info("Данные для графиков успешно загружены.")

        fig = px.line(
            df_loss,
            x="Epoch",
            y="Loss Value",
            color="Loss Type",
            title="Потери",
            template=theme_model,
        )
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
            ),
            yaxis=dict(
                showgrid=True,
            ),
            title=dict(
                x=0.4,
            ),
        )
        fig.update_layout(
            xaxis_title="Эпохи",
            yaxis_title="Значения потерь",
            legend=dict(title="Нейросеть"),
        )
        fig.update_traces(name="Дискриминатор", selector=dict(name="D_loss"))
        fig.update_traces(name="Генератор", selector=dict(name="G_loss"))
        st.plotly_chart(fig)
        logger.info("График потерь успешно создан и отображен.")

        st.header("Метрика FID")
        fig = px.line(
            df_metric,
            x="Epoch",
            y="FID score",
            title="Метрика FID",
            template=theme_model,
        )
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
            ),
            yaxis=dict(
                showgrid=True,
            ),
            title=dict(
                x=0.4,
            ),
        )
        fig.update_layout(
            xaxis_title="Эпохи",
            yaxis_title="Значения метрики",
        )
        st.plotly_chart(fig)

with st.expander("Загрузка данных: свернуть/развернуть", expanded=True):
    st.header("Загрузка данных")
    example = {
        "image": ["{'bytes': b'\\x89PNG\\r\\n\\x1a\\n\\x00\\...'}"],
        "text": ["Simple elegant logo for Concept, love orange ..."],
    }
    example = pd.DataFrame(example)
    example.index = range(456, 457)
    st.markdown(f"Требования к датасету")
    st.write(example)
    example_dataset = "https://drive.google.com/file/d/1BiUi9TOVgIjEggFQHb9d49Dp-z0pgIvI/view?usp=sharing"
    st.markdown(
        "Формат `parquet`. "
        "Изображения представлены в байтовом виде внутри словаря, "
        "текст представлен в виде обычной строки с перечислением эпитетов. "
        f"Ссылка на [пример датасета]({example_dataset})."
    )

    uploaded_file = st.file_uploader("Загрузите датасет", type=["parquet"])
    if uploaded_file is not None:
        dataset = pd.read_parquet(uploaded_file)
        dataset["image"] = dataset["image"].apply(
            lambda x: Image.open(BytesIO(x.get("bytes")))
        )
        st.markdown(
            f'<span style="color:gray">Количество объектов в датасете: </span>{len(dataset)}',
            unsafe_allow_html=True,
        )
if uploaded_file is not None:
    dataset = pd.read_parquet(uploaded_file)
    try:
        dataset["image"] = dataset["image"].apply(
            lambda x: Image.open(BytesIO(x.get("bytes")))
        )
    except Exception as e:
        logger.error(f"Ошибка извлечения изображения: {e}")
        st.error(f"Ошибка извлечения изображения: {e}")
        

    with st.expander(
        "Получить случайный элемент датасета: свернуть/развернуть", expanded=True
    ):
        if st.button("Получить случайный элемент датасета"):
            ind = random.randint(0, len(dataset) + 1)
            st.session_state.index = ind
            st.session_state.dataset_image = dataset["image"][ind]
            st.session_state.dataset_text = dataset["text"][ind]
            logger.info("Получен случайный элемент датасета: индекс %d", ind)

        if "dataset_image" in st.session_state:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    st.session_state.dataset_image,
                    caption=f"Логотип из датасета: индекс {st.session_state.index}",
                )
            st.markdown(
                f'<span style="color:gray">Текстовое описание логотипа: </span>{st.session_state.dataset_text}',
                unsafe_allow_html=True,
            )

    with st.expander("EDA: свернуть/развернуть", expanded=True):
        if st.button("EDA"):
            st.session_state.eda_clicked = True
            try:
                dataset_for_eda = dataset_to_eda(dataset)
                st.session_state.dataset_for_EDA = dataset_for_eda
                logger.info("EDA успешно выполнен, данные подготовлены для анализа.")
            except Exception as e:
                st.error(f"Ошибка данных датасета: {e}")
                logger.error(f"Ошибка форматирования датасета: {e}")

        if st.session_state.eda_clicked:
            if st.session_state.dataset_for_EDA is not None:
                data = st.session_state.dataset_for_EDA
                st.title("Анализ текстовых данных")
                st.subheader("Облако слов")
                if "word_cloud" not in st.session_state:
                    wc = WordCloud(background_color="black", width=1000, height=500)
                    words = data["description"].explode().values
                    wc.generate(" ".join(words))

                    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    ax.set_title(f"Облако слов для описаний логотипов", fontsize=30)
                    ax.imshow(wc, alpha=0.98)

                    st.session_state.word_cloud = fig
                st.pyplot(st.session_state.word_cloud)

                theme = st.selectbox(
                    "Выберите тему для графиков", sorted(themes), key="theme_selector"
                )

                st.subheader("Гистограмма распределения длин описаний")
                bins = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_description"
                )
                bin_edges = np.linspace(data["len"].min(), data["len"].max(), bins + 1)
                fig = px.histogram(
                    data,
                    x="len",
                    nbins=bins,
                    title="Распределение длин описаний",
                    template=theme,
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges[0],
                        end=bin_edges[-1],
                        size=(bin_edges[1] - bin_edges[0]),
                    ),
                )
                logger.info("Гистограмма распределения длин описаний успешно создана и отображена.")

                fig.update_layout(
                    bargap=0.1,
                    xaxis=dict(
                        showgrid=True,
                    ),
                    yaxis=dict(
                        showgrid=True,
                    ),
                    title=dict(
                        x=0.4,
                    ),
                )
                fig.update_layout(
                    xaxis_title="Длина описания в символах",
                    yaxis_title="Частота",
                )
                st.plotly_chart(fig)
                logger.info("Гистограмма распределения длин описаний успешно создана и отображена.")
                st.error(f"Ошибка при создании гистограммы: {e}")

                st.subheader("Boxplot для длин описаний")
                fig = px.box(data, x="len", template=theme)
                fig.update_layout(
                    title=dict(
                        text="Boxplot длин описаний",
                        x=0.4,
                    ),
                    xaxis=dict(
                        showgrid=True,
                        title="Длина описания в символах",
                    ),
                )
                st.plotly_chart(fig)

                st.subheader("Гистограмма количества эпитетов в описании*")
                st.write(
                    "*количество слов и словосочетаний, записанных через запятую, в описании изображения"
                )
                bins6 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_epitets_num"
                )
                bin_edges6 = np.linspace(
                    data["epitets_num"].min(), data["epitets_num"].max(), bins6 + 1
                )
                fig = px.histogram(
                    data,
                    x="epitets_num",
                    nbins=bins6,
                    title="Количество эпитетов в описании",
                    template=theme,
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges6[0],
                        end=bin_edges6[-1],
                        size=(bin_edges6[1] - bin_edges6[0]),
                    ),
                )
                fig.update_layout(
                    bargap=0.1,
                    xaxis=dict(
                        showgrid=True,
                    ),
                    yaxis=dict(
                        showgrid=True,
                    ),
                    title=dict(
                        x=0.4,
                    ),
                )
                fig.update_layout(
                    xaxis_title="Количество эпитетов",
                    yaxis_title="Частота",
                )
                st.plotly_chart(fig)

                st.subheader("Boxplot для количества эпитетов")
                fig = px.box(data, x="epitets_num", template=theme)
                fig.update_layout(
                    title=dict(
                        text="Количество эпитетов в описании",
                        x=0.4,
                    ),
                    xaxis=dict(
                        showgrid=True,
                        title="Количество эпитетов",
                    ),
                )
                st.plotly_chart(fig)

                st.title("Анализ изображений")
                value_counts = data["rgb"].value_counts()
                pie_data = pd.DataFrame(
                    {
                        "Value": ["RGB", "BW"],
                        "Count": [
                            value_counts.get("RGB", 0),
                            value_counts.get("BW", 0),
                        ],
                    }
                )
                st.subheader("Соотношение RGB и чёрно-белых логотипов")
                fig = px.pie(pie_data, names="Value", values="Count", template=theme)

                fig.update_layout(
                    title=dict(
                        text="Тип изображения",
                        x=0.4,
                    )
                )
                st.plotly_chart(fig)

                st.subheader("Высота изображений")
                bins1 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_height"
                )
                bin_edges1 = np.linspace(data["h"].min(), data["h"].max(), bins1 + 1)
                fig = px.histogram(
                    data,
                    x="h",
                    nbins=bins1,
                    title="Распределение высоты изображений",
                    template=theme,
                )
                fig.update_layout(
                    xaxis_title="Высота",
                    yaxis_title="Частота",
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges1[0],
                        end=bin_edges1[-1],
                        size=(bin_edges1[1] - bin_edges1[0]),
                    ),
                )
                fig.update_layout(
                    title=dict(
                        text="Высота изображений",
                        x=0.4,
                    )
                )
                st.plotly_chart(fig)

                st.subheader("Ширина изображений")
                bins2 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_wight"
                )
                bin_edges2 = np.linspace(data["w"].min(), data["w"].max(), bins2 + 1)
                fig = px.histogram(data, x="h", nbins=bins2, template=theme)
                fig.update_layout(
                    xaxis_title="Ширина",
                    yaxis_title="Частота",
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges2[0],
                        end=bin_edges2[-1],
                        size=(bin_edges2[1] - bin_edges2[0]),
                    ),
                )
                fig.update_layout(
                    title=dict(
                        text="Ширина изображений",
                        x=0.4,
                    )
                )
                st.plotly_chart(fig)

                st.subheader("Соотношение сторон")
                bins3 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_ratio"
                )
                bin_edges3 = np.linspace(
                    data["ratio"].min(), data["ratio"].max(), bins3 + 1
                )
                fig = px.histogram(data, x="ratio", nbins=bins3, template=theme)
                fig.update_layout(
                    xaxis_title="Соотношение сторон",
                    yaxis_title="Частота",
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges3[0],
                        end=bin_edges3[-1],
                        size=(bin_edges3[1] - bin_edges3[0]),
                    ),
                )
                fig.update_layout(
                    title=dict(
                        text="Распределение соотношений сторон h/w",
                        x=0.4,
                    )
                )
                st.plotly_chart(fig)

                st.subheader("Соотношение сторон (без квадратных изображений)")
                bins4 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_ratio1"
                )
                bin_edges4 = np.linspace(
                    data[data["ratio"] != 1]["ratio"].min(),
                    data[data["ratio"] != 1]["ratio"].max(),
                    bins4 + 1,
                )
                fig = px.histogram(
                    data[data["ratio"] != 1]["ratio"],
                    x="ratio",
                    nbins=bins4,
                    template=theme,
                )
                fig.update_layout(
                    xaxis_title="Соотношение сторон",
                    yaxis_title="Частота",
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges4[0],
                        end=bin_edges4[-1],
                        size=(bin_edges4[1] - bin_edges4[0]),
                    ),
                )
                fig.update_layout(
                    title=dict(
                        text="Распределение соотношений сторон h/w",
                        x=0.4,
                    )
                )
                st.plotly_chart(fig)

                st.subheader("Количество пикселей")
                bins5 = st.slider(
                    "Количество интервалов (bins)", 5, 50, 10, key="bins_pixel"
                )
                bin_edges5 = np.linspace(
                    data["pixel"].min(), data["pixel"].max(), bins5 + 1
                )
                fig = px.histogram(data, x="pixel", nbins=bins5, template=theme)
                fig.update_layout(
                    xaxis_title="Количество пикселей",
                    yaxis_title="Частота",
                )
                fig.update_traces(
                    xbins=dict(
                        start=bin_edges5[0],
                        end=bin_edges5[-1],
                        size=(bin_edges5[1] - bin_edges5[0]),
                    ),
                )
                fig.update_layout(
                    title=dict(
                        text="Распределение количества пикселей",
                        x=0.4,
                    )
                )
                st.plotly_chart(fig)
