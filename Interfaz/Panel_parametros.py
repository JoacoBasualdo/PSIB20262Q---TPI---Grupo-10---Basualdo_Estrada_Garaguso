import inspect
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from Funciones import Lectura
from main import RUTA_DATASET, RUTA_DATOS_CLINICOS, RUTA_MRBRAINS, ejecutar_pipeline

VALORES_POR_DEFECTO = {
    nombre: parametro.default
    for nombre, parametro in inspect.signature(ejecutar_pipeline).parameters.items()
}

GRUPOS = {
    "Nondemented": "Sano",
    "Demented": "Con demencia",
    "Converted": "Convertido (sano → demencia)",
}


def hay_mrbrains():
    return os.path.isdir(RUTA_MRBRAINS) and any(
        "segm.nii.gz" in archivos for _, _, archivos in os.walk(RUTA_MRBRAINS)
    )


class PanelParametros(QWidget):

    ejecutar_solicitado = Signal(dict)

    def __init__(self):
        super().__init__()

        self.pacientes = Lectura.listar_pacientes(RUTA_DATASET, RUTA_DATOS_CLINICOS)
        self.mrbrains_disponible = hay_mrbrains()

        disposicion = QVBoxLayout(self)
        disposicion.addWidget(self._crear_grupo_paciente())
        disposicion.addWidget(self._crear_grupo_validacion())
        disposicion.addWidget(self._crear_boton_avanzadas())
        disposicion.addWidget(self._crear_grupo_avanzadas())
        disposicion.addStretch()

        self.boton_ejecutar = QPushButton("Ejecutar")
        self.boton_ejecutar.setObjectName("botonPrincipal")
        self.boton_ejecutar.setCursor(Qt.PointingHandCursor)
        self.boton_ejecutar.clicked.connect(self._al_ejecutar)
        disposicion.addWidget(self.boton_ejecutar)

        self._restaurar_valores()
        self._al_cambiar_paciente()

    def _crear_grupo_paciente(self):
        grupo = QGroupBox("Paciente")
        formulario = QFormLayout(grupo)

        self.combo_paciente = QComboBox()
        self.combo_paciente.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_paciente.setMinimumContentsLength(14)
        for paciente_id, info in self.pacientes.items():
            self.combo_paciente.addItem(f"{paciente_id}  ·  {GRUPOS.get(info['grupo'], info['grupo'])}", paciente_id)
        self.combo_paciente.currentIndexChanged.connect(self._al_cambiar_paciente)
        formulario.addRow("Paciente:", self.combo_paciente)

        self.campo_visitas = QSpinBox()
        self.campo_visitas.valueChanged.connect(self._al_cambiar_visitas)
        formulario.addRow("Visitas a procesar:", self.campo_visitas)

        self.etiqueta_info = QLabel()
        self.etiqueta_info.setWordWrap(True)
        formulario.addRow(self.etiqueta_info)

        return grupo

    def _crear_grupo_validacion(self):
        grupo = QGroupBox("Validación")
        formulario = QFormLayout(grupo)

        self.casilla_oasis = QCheckBox("Concordancia con OASIS")
        self.casilla_oasis.setToolTip("Compara la tasa de atrofia propia con el nWBV publicado por OASIS. Tarda segundos.")
        formulario.addRow(self.casilla_oasis)

        self.casilla_test_retest = QCheckBox("Test-retest (lento)")
        self.casilla_test_retest.setToolTip("Mide mpr-1 y mpr-2 de una misma sesión por separado: la diferencia es el error del método. Tarda 2-3 min por visita.")
        formulario.addRow(self.casilla_test_retest)

        self.casilla_mrbrains = QCheckBox("MRBrainS18 (lento)")
        self.casilla_mrbrains.setToolTip("Compara los ventrículos contra segmentaciones manuales de expertos. Tarda 1-2 min por sujeto.")
        if not self.mrbrains_disponible:
            self.casilla_mrbrains.setEnabled(False)
            self.casilla_mrbrains.setText("MRBrainS18 (no descargado)")
            self.casilla_mrbrains.setToolTip("Para habilitarla, descomprimir training.zip en Dataset/MRBrainS18/.")
        formulario.addRow(self.casilla_mrbrains)

        return grupo

    def _crear_boton_avanzadas(self):
        self.boton_avanzadas = QToolButton()
        self.boton_avanzadas.setObjectName("botonPlegable")
        self.boton_avanzadas.setCursor(Qt.PointingHandCursor)
        self.boton_avanzadas.setText("Opciones avanzadas")
        self.boton_avanzadas.setCheckable(True)
        self.boton_avanzadas.setArrowType(Qt.RightArrow)
        self.boton_avanzadas.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.boton_avanzadas.toggled.connect(self._al_desplegar_avanzadas)
        return self.boton_avanzadas

    def _crear_grupo_avanzadas(self):
        self.grupo_avanzadas = QGroupBox()
        self.grupo_avanzadas.setObjectName("grupoAvanzadas")
        formulario = QFormLayout(self.grupo_avanzadas)

        self.campo_visitas_test_retest = QSpinBox()
        self.campo_visitas_test_retest.setRange(1, 5)
        formulario.addRow("Visitas para test-retest:", self.campo_visitas_test_retest)

        boton_restaurar = QPushButton("Restaurar valores por defecto")
        boton_restaurar.clicked.connect(self._restaurar_valores)
        formulario.addRow(boton_restaurar)

        self.grupo_avanzadas.setVisible(False)
        return self.grupo_avanzadas

    def _restaurar_valores(self):
        v = VALORES_POR_DEFECTO
        self.casilla_oasis.setChecked(v["validar_concordancia_oasis"])
        self.casilla_test_retest.setChecked(v["validar_test_retest"])
        self.casilla_mrbrains.setChecked(v["validar_mrbrains"] and self.mrbrains_disponible)
        self.campo_visitas_test_retest.setValue(v["visitas_test_retest"])

    def _al_desplegar_avanzadas(self, desplegado):
        self.grupo_avanzadas.setVisible(desplegado)
        self.boton_avanzadas.setArrowType(Qt.DownArrow if desplegado else Qt.RightArrow)

    def _al_cambiar_paciente(self):
        paciente_id = self.combo_paciente.currentData()
        hay_paciente = paciente_id is not None
        self.boton_ejecutar.setEnabled(hay_paciente)
        self.campo_visitas.setEnabled(hay_paciente)

        if not hay_paciente:
            self.etiqueta_info.setText(
                "No se encontraron pacientes en Dataset/. Correr primero "
                "«python descargar_dataset.py» desde la carpeta del proyecto."
            )
            return

        info = self.pacientes[paciente_id]
        self.campo_visitas.setRange(1, info["visitas_excel"])
        self.campo_visitas.setValue(min(VALORES_POR_DEFECTO["cantidad_visitas"], info["visitas_excel"]))

        cdr = " → ".join(f"{c:g}" for c in info["cdr"])
        mmse = " → ".join(f"{m:g}" for m in info["mmse"])
        aviso = ""
        if info["visitas_en_disco"] < info["visitas_excel"]:
            aviso = (f"\nAtención: hay {info['visitas_en_disco']} de {info['visitas_excel']} visitas en el disco; "
                     "las que falten se saltean.")
        self.etiqueta_info.setText(
            f"Grupo: {GRUPOS.get(info['grupo'], info['grupo'])}\n"
            f"Edad: {info['edad_inicial']} a {info['edad_final']} años\n"
            f"CDR: {cdr}\n"
            f"MMSE: {mmse}"
            f"{aviso}"
        )
        self._al_cambiar_visitas(self.campo_visitas.value())

    def _al_cambiar_visitas(self, cantidad):
        posible = cantidad >= 2
        self.casilla_oasis.setEnabled(posible)
        if not posible:
            self.casilla_oasis.setChecked(False)
        self.campo_visitas_test_retest.setMaximum(max(cantidad, 1))

    def obtener_parametros(self):
        return {
            "paciente_id": self.combo_paciente.currentData(),
            "cantidad_visitas": self.campo_visitas.value(),
            "validar_concordancia_oasis": self.casilla_oasis.isChecked(),
            "validar_test_retest": self.casilla_test_retest.isChecked(),
            "visitas_test_retest": self.campo_visitas_test_retest.value(),
            "validar_mrbrains": self.casilla_mrbrains.isChecked(),
        }

    def _validar(self, parametros):
        errores = []
        if parametros["paciente_id"] is None:
            errores.append("Elegí un paciente.")
        if parametros["validar_test_retest"] and parametros["visitas_test_retest"] > parametros["cantidad_visitas"]:
            errores.append("Las visitas de test-retest no pueden ser más que las visitas a procesar.")
        return errores

    def _al_ejecutar(self):
        parametros = self.obtener_parametros()
        errores = self._validar(parametros)
        if errores:
            QMessageBox.warning(self, "Parámetros inválidos", "\n".join(errores))
            return
        self.ejecutar_solicitado.emit(parametros)