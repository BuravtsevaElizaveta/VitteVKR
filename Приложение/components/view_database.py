# components/view_database.py

import streamlit as st
import pandas as pd
from sqlalchemy.orm import Session
from utils.database import Detection
import logging
import hashlib

logger = logging.getLogger(__name__)

# Задайте хэшированное значение пароля для защиты функции очистки
# В реальном приложении используйте безопасные методы хранения паролей
PASSWORD_HASH = hashlib.sha256("ваш_секретный_пароль".encode()).hexdigest()

def render_database_view(db_session: Session):
    """
    Отрисовывает таблицу с детекциями из базы данных и предоставляет возможность очистки базы.
    
    Параметры:
    - db_session (Session): Сессия SQLAlchemy для доступа к базе данных.
    """
    try:
        st.header("Просмотр базы данных детекций")
        
        # Инициализация состояния подтверждения очистки
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False
        
        # Запрос всех детекций из базы данных
        detections = db_session.query(Detection).order_by(Detection.timestamp.desc()).all()
        
        if detections:
            # Преобразование в DataFrame для удобства отображения
            data = {
                "ID": [det.id for det in detections],
                "Время": [det.timestamp.strftime("%Y-%m-%d %H:%M:%S") for det in detections],
                "Класс объекта": [det.object_class for det in detections],
                "Количество": [det.quantity for det in detections]
            }
            df = pd.DataFrame(data)
            
            # Добавление возможности скачивания таблицы в CSV
            st.download_button(
                label="Скачать детекции как CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='detections.csv',
                mime='text/csv',
            )
            
            # Отображение таблицы с детекциями
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
            
            # Кнопка для очистки базы данных
            if not st.session_state.confirm_clear:
                if st.button("Очистить базу данных"):
                    st.session_state.confirm_clear = True
            else:
                st.warning("Для подтверждения очистки базы данных, введите пароль.")
                password = st.text_input("Пароль", type="password")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Да, очистить"):
                        if hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
                            try:
                                deleted = db_session.query(Detection).delete()
                                db_session.commit()
                                logger.info(f"База данных детекций очищена. Удалено записей: {deleted}")
                                st.success("База данных успешно очищена.")
                                st.session_state.confirm_clear = False
                            except Exception as e:
                                db_session.rollback()
                                logger.error(f"Ошибка при очистке базы данных: {e}")
                                st.error("Не удалось очистить базу данных.")
                                st.session_state.confirm_clear = False
                        else:
                            st.error("Неверный пароль.")
                            st.session_state.confirm_clear = False
                with col2:
                    if st.button("Отмена"):
                        st.session_state.confirm_clear = False
        else:
            st.info("В базе данных нет детекций.")
            
            # Если база данных пустая, сбросить состояние подтверждения
            if 'confirm_clear' not in st.session_state:
                st.session_state.confirm_clear = False
            
            if st.session_state.confirm_clear:
                st.warning("База данных уже пустая.")
                st.session_state.confirm_clear = False
    except Exception as e:
        logger.error(f"Ошибка при отображении базы данных: {e}")
        st.error("Не удалось загрузить данные из базы данных.")
