FROM python:3.12-slim

# TeX Live minimo: latex-recommended + fonts-recommended traen mathptmx (Times New
# Roman) y pdflatex, que es todo lo que el reporte de 2 columnas necesita.
# texlive-full son ~4 GB que no se usan.
RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-fonts-recommended \
        lmodern \
        poppler-utils \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /lab

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# El codigo vive en /lab/src y se importa como paquete desde el notebook.
ENV PYTHONPATH=/lab
ENV MPLBACKEND=Agg
# Sin buffer: el benchmark corre decenas de minutos y su progreso tiene que verse
# en los logs mientras avanza, no todo junto al final.
ENV PYTHONUNBUFFERED=1
ENV LAB_DATA_DIR=/lab/archivos

EXPOSE 8888

CMD ["bash"]
