from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

ZOOM_MINIMO = 0.25
ZOOM_MAXIMO = 3.0
PASO_ZOOM = 1.25


class ImagenAjustable(QLabel):

    def __init__(self, imagen):
        super().__init__()
        self.original = imagen
        self.setAlignment(Qt.AlignLeft)
        self.setPixmap(imagen)

    def ajustar_ancho(self, ancho_disponible, zoom):
        ancho_base = min(self.original.width(), max(ancho_disponible, 200))
        ancho = max(int(ancho_base * zoom), 50)
        self.setPixmap(self.original.scaledToWidth(ancho, Qt.SmoothTransformation))


class AreaDesplazable(QScrollArea):

    def __init__(self, al_hacer_zoom, al_redimensionar):
        super().__init__()
        self.al_hacer_zoom = al_hacer_zoom
        self.al_redimensionar = al_redimensionar

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        self.al_redimensionar()

    def wheelEvent(self, evento):
        if evento.modifiers() & Qt.ControlModifier:
            self.al_hacer_zoom(PASO_ZOOM if evento.angleDelta().y() > 0 else 1 / PASO_ZOOM)
            evento.accept()
        else:
            super().wheelEvent(evento)


class PestanaEtapa(QWidget):

    def __init__(self, titulo, elementos):
        super().__init__()
        self.imagenes = []
        self.zoom = 1.0

        disposicion = QVBoxLayout(self)
        disposicion.setContentsMargins(0, 0, 0, 0)
        disposicion.addLayout(self._crear_barra_zoom())

        self.area = AreaDesplazable(self._multiplicar_zoom, self._ajustar_imagenes)
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QScrollArea.NoFrame)
        disposicion.addWidget(self.area)

        contenido = QWidget()
        disposicion_contenido = QVBoxLayout(contenido)

        etiqueta_titulo = QLabel(titulo)
        fuente_titulo = etiqueta_titulo.font()
        fuente_titulo.setPointSize(14)
        fuente_titulo.setBold(True)
        etiqueta_titulo.setFont(fuente_titulo)
        etiqueta_titulo.setStyleSheet("color: #02568A; padding-bottom: 6px;")
        etiqueta_titulo.setWordWrap(True)
        disposicion_contenido.addWidget(etiqueta_titulo)

        if not elementos:
            disposicion_contenido.addWidget(QLabel("Esta etapa no generó resultados con las opciones elegidas."))

        for tipo, valor in elementos:
            if tipo == "subtitulo":
                disposicion_contenido.addWidget(self._crear_subtitulo(valor))
            elif tipo == "texto":
                disposicion_contenido.addWidget(self._crear_texto(valor))
            elif tipo == "figura":
                disposicion_contenido.addWidget(self._crear_imagen(valor))

        disposicion_contenido.addStretch()
        self.area.setWidget(contenido)

    def _crear_barra_zoom(self):
        barra = QHBoxLayout()
        barra.setContentsMargins(8, 6, 8, 2)
        barra.addStretch()

        self.etiqueta_zoom = QLabel("100 %")
        self.etiqueta_zoom.setMinimumWidth(48)
        self.etiqueta_zoom.setAlignment(Qt.AlignCenter)

        boton_alejar = QPushButton("−")
        boton_alejar.setToolTip("Alejar (Ctrl + rueda del mouse)")
        boton_alejar.clicked.connect(lambda: self._multiplicar_zoom(1 / PASO_ZOOM))

        boton_acercar = QPushButton("+")
        boton_acercar.setToolTip("Acercar (Ctrl + rueda del mouse)")
        boton_acercar.clicked.connect(lambda: self._multiplicar_zoom(PASO_ZOOM))

        boton_ajustar = QPushButton("Ajustar al ancho")
        boton_ajustar.clicked.connect(lambda: self._fijar_zoom(1.0))

        for boton in (boton_alejar, boton_acercar):
            boton.setFixedWidth(36)
        for widget in (QLabel("Zoom de las figuras:"), boton_alejar, self.etiqueta_zoom, boton_acercar, boton_ajustar):
            barra.addWidget(widget)
        return barra

    def _crear_subtitulo(self, texto):
        etiqueta = QLabel(texto)
        fuente = etiqueta.font()
        fuente.setPointSize(11)
        fuente.setBold(True)
        etiqueta.setFont(fuente)
        etiqueta.setStyleSheet("color: #01426B; border-bottom: 2px solid #D5DEE8; padding: 10px 0 4px 0;")
        return etiqueta

    def _crear_texto(self, texto):
        etiqueta = QLabel(texto)
        etiqueta.setTextFormat(Qt.PlainText)
        etiqueta.setFont(QFont("Consolas", 9))
        etiqueta.setTextInteractionFlags(Qt.TextSelectableByMouse)
        etiqueta.setStyleSheet("background: #F7F9FC; padding: 8px;")

        contenedor = QScrollArea()
        contenedor.setWidget(etiqueta)
        contenedor.setFrameShape(QScrollArea.NoFrame)
        contenedor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        contenedor.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        contenedor.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        contenedor.setFixedHeight(etiqueta.sizeHint().height() + contenedor.horizontalScrollBar().sizeHint().height() + 4)
        contenedor.setStyleSheet("QScrollArea { background: #F7F9FC; border-radius: 6px; }")
        return contenedor

    def _crear_imagen(self, datos_png):
        imagen = QPixmap()
        imagen.loadFromData(datos_png, "PNG")
        etiqueta = ImagenAjustable(imagen)
        self.imagenes.append(etiqueta)
        return etiqueta

    def _multiplicar_zoom(self, factor):
        self._fijar_zoom(self.zoom * factor)

    def _fijar_zoom(self, zoom):
        self.zoom = min(max(zoom, ZOOM_MINIMO), ZOOM_MAXIMO)
        self.etiqueta_zoom.setText(f"{self.zoom * 100:.0f} %")
        self._ajustar_imagenes()

    def _ajustar_imagenes(self):
        if not hasattr(self, "area"):
            return
        ancho = self.area.viewport().width() - 30
        for imagen in self.imagenes:
            imagen.ajustar_ancho(ancho, self.zoom)