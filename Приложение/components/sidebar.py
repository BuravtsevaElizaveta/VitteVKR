# components/sidebar.py

import streamlit as st
import torch
import logging

logger = logging.getLogger(__name__)

def render_sidebar():
    """
    Отрисовывает боковую панель с настройками.

    Возвращает:
    - model_option (str): Выбранная модель ('YOLOv8' или 'YOLOv9').
    - model_path (str): Путь к файлу весов модели.
    - device_option (str): Выбранное устройство ('CPU' или 'GPU').
    - conf_thresh (float): Порог уверенности для детекций.
    - line_thickness (int): Толщина линий для bounding boxes.
    """
    st.sidebar.header("Настройки")
    
    # Раздел выбора модели
    st.sidebar.subheader("Выбор модели")
    model_option = st.sidebar.selectbox("Выберите модель YOLO", ["YOLOv8", "YOLOv9"])
    model_path = st.sidebar.text_input(f"Путь к модели {model_option}", f"Приложение/models/{model_option}.pt")
    
    # Раздел выбора устройства
    st.sidebar.subheader("Опции процессора")
    device_option = st.sidebar.radio("Выберите устройство", ["CPU", "GPU"], index=0)
    
    # Раздел настроек детекции
    st.sidebar.subheader("Настройки детекции")
    conf_thresh = st.sidebar.slider("Порог уверенности", 0.0, 1.0, 0.25, 0.05)
    line_thickness = st.sidebar.slider("Толщина линии", 1, 10, 2)
    
    # Проверка доступности CUDA
    if device_option == "GPU":
        if not torch.cuda.is_available():
            st.sidebar.warning("CUDA недоступна, выберите CPU")
            device_option = "CPU"
            logger.warning("CUDA недоступна, переключение на CPU.")
        else:
            logger.info("CUDA доступна, используется GPU.")
    else:
        logger.info("Используется CPU.")
    
    return model_option, model_path, device_option, conf_thresh, line_thickness

