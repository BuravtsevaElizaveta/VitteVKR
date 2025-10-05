# components/statistics.py

import streamlit as st
from utils.system_stats import get_system_stats, get_disk_usage
import logging
import plotly.express as px

logger = logging.getLogger(__name__)

def render_system_stats():
    """
    Отрисовывает системную статистику в главном окне (вкладка «Статистика»).
    """
    try:
        st.subheader("Системная статистика")

        stats = get_system_stats()
        cpu = stats.get("cpu_percent", 0)
        mem = stats.get("memory_percent", 0)
        st.text(f"CPU: {cpu:.1f}%")
        st.text(f"Memory: {mem:.1f}%")

        disk = get_disk_usage("/")  # или путь к корню
        st.text(f"Disk: {disk['percent']:.1f}% ({disk['used_gb']:.1f} / {disk['total_gb']:.1f} GB)")

        st.markdown("---")
        st.subheader("График загрузки ресурсов")
        fig = px.pie(
            names=["CPU", "Memory", "Disk"],
            values=[cpu, mem, disk["percent"]],
            title="Использование ресурсов"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Ошибка при отображении системной статистики: {e}")
        st.error("Не удалось загрузить статистику системы.")
