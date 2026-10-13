"""Material didáctico interactivo para un curso de Diseño Experimental."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats


st.set_page_config(
    page_title="Diseño Experimental | Aula interactiva",
    page_icon="🧪",
    layout="wide",
)

# Metadatos que se reutilizan en la navegación y en la página de inicio.
DISENOS = {
    "Inicio": {
        "icono": "🏠",
        "resumen": "Objetivos del curso y diseños disponibles.",
    },
    "DCA": {
        "icono": "🎲",
        "resumen": "Un factor con asignación completamente aleatoria.",
    },
    "DBCA": {
        "icono": "🧱",
        "resumen": "Tratamientos comparados dentro de bloques homogéneos.",
    },
    "DCL": {
        "icono": "🔲",
        "resumen": "Control local simultáneo en dos direcciones.",
    },
    "DPD": {
        "icono": "🌱",
        "resumen": "Dos factores aplicados en parcelas de distinto tamaño.",
    },
    "Diseños Factoriales": {
        "icono": "⚗️",
        "resumen": "Efectos principales e interacciones entre factores.",
    },
    "MSR": {
        "icono": "🏔️",
        "resumen": "Modelado y optimización de respuestas continuas.",
    },
}

COLORES = px.colors.qualitative.Safe


def mostrar_ejercicio(pregunta, opciones, respuesta, explicacion):
    """Presenta una pregunta autocorregible para el estudiante."""
    st.subheader("📝 Comprueba lo aprendido")
    seleccion = st.radio(pregunta, opciones, key=f"ejercicio_{pregunta}")
    if st.button("Revisar respuesta", key=f"revisar_{pregunta}"):
        if seleccion == respuesta:
            st.success(f"¡Correcto! {explicacion}")
        else:
            st.error(f"No exactamente. {explicacion}")


def mostrar_anova(modelo, tipo=2):
    """Muestra ANOVA de un modelo lineal usando una tabla clara."""
    tabla = sm.stats.anova_lm(modelo, typ=tipo)
    tabla.index.name = "Fuente de variación"
    tabla = tabla.rename(
        columns={
            "df": "Grados de libertad",
            "sum_sq": "Suma de cuadrados",
            "mean_sq": "Cuadrado medio",
            "F": "F",
            "PR(>F)": "p-valor",
        }
    )
    st.dataframe(tabla.round(4), width="stretch")


def mostrar_tabla_grados(fuentes):
    """Muestra las fuentes de variación y sus grados de libertad."""
    st.dataframe(
        pd.DataFrame(fuentes, columns=["Fuente de variación", "Grados de libertad"]),
        hide_index=True,
        width="stretch",
    )


def pagina_inicio():
    st.title("🧪 Diseño Experimental")
    st.subheader("Aula interactiva para aprender, analizar y visualizar")
    st.write(
        "Esta aplicación acompaña un curso de Diseño Experimental con ejemplos "
        "pequeños y reproducibles. En cada módulo se presenta la lógica del diseño, "
        "su modelo estadístico, una tabla ANOVA, datos de ejemplo y una visualización."
    )
    st.write(
        "El objetivo es conectar la planificación del experimento con el análisis "
        "de resultados: identificar las fuentes de variación, elegir el término "
        "de error apropiado y comunicar las conclusiones con apoyo gráfico."
    )
    st.markdown("### Diseños disponibles")
    columnas = st.columns(3)
    for indice, (nombre, info) in enumerate(list(DISENOS.items())[1:]):
        with columnas[indice % 3]:
            st.markdown(f"#### {info['icono']} {nombre}")
            st.write(info["resumen"])
    st.info(
        "Selecciona un diseño en la barra lateral para explorar sus supuestos, "
        "ejemplos y preguntas de práctica."
    )
    mostrar_ejercicio(
        "¿Cuál es el propósito principal de bloquear unidades experimentales?",
        [
            "Aumentar el número de tratamientos",
            "Controlar una fuente conocida de variación no deseada",
            "Eliminar la necesidad de aleatorizar",
        ],
        "Controlar una fuente conocida de variación no deseada",
        "Los bloques agrupan unidades similares y ayudan a separar esa variabilidad "
        "del efecto de los tratamientos.",
    )


def pagina_dca():
    st.title("🎲 DCA — Diseño Completamente al Azar")
    st.write(
        "En el DCA, los tratamientos se asignan al azar a unidades experimentales "
        "que se consideran homogéneas. Es el diseño más sencillo y funciona bien "
        "cuando no hay una fuente importante de variación que convenga controlar."
    )
    st.write(
        "La comparación se realiza mediante un análisis de varianza de un factor. "
        "La aleatorización protege frente a sesgos sistemáticos, mientras que la "
        "replicación permite estimar el error experimental y la precisión de las medias."
    )
    st.latex(r"Y_{ij} = \mu + \tau_i + \varepsilon_{ij}")
    st.caption(
        r"$Y_{ij}$: observación; $\mu$: media general; $\tau_i$: efecto del tratamiento; "
        r"$\varepsilon_{ij}$: error aleatorio independiente."
    )

    dietas = {
        "Dieta A": [4.8, 5.1, 4.9, 5.3, 4.7],
        "Dieta B": [5.6, 5.8, 5.4, 5.9, 5.7],
        "Dieta C": [6.2, 6.0, 6.5, 6.3, 6.1],
    }
    datos = pd.DataFrame(
        [
            {"Dieta": dieta, "Ganancia (kg)": valor}
            for dieta, valores in dietas.items()
            for valor in valores
        ]
    )
    st.markdown("### Ejemplo: ganancia de peso con tres dietas")
    st.write(
        "Se asignan al azar 15 animales a tres dietas y se registra la ganancia "
        "de peso al final del ensayo."
    )
    st.dataframe(datos, hide_index=True, width="stretch")
    modelo = smf.ols('Q("Ganancia (kg)") ~ C(Dieta)', data=datos).fit()
    st.markdown("#### ANOVA de un factor")
    mostrar_anova(modelo)
    st.caption("En un DCA con a tratamientos y N observaciones: tratamientos a−1, error N−a, total N−1.")

    resumen = datos.groupby("Dieta")["Ganancia (kg)"].agg(["mean", "std", "count"])
    critico = stats.t.ppf(0.975, resumen["count"] - 1)
    resumen["inferior"] = resumen["mean"] - critico * resumen["std"] / np.sqrt(resumen["count"])
    resumen["superior"] = resumen["mean"] + critico * resumen["std"] / np.sqrt(resumen["count"])
    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=resumen.index,
            y=resumen["mean"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": resumen["superior"] - resumen["mean"],
                "arrayminus": resumen["mean"] - resumen["inferior"],
            },
            mode="markers",
            marker={"size": 13, "color": COLORES[0]},
            name="Media e IC 95 %",
        )
    )
    figura.update_layout(
        title="Ganancia media por dieta con intervalos de confianza del 95 %",
        xaxis_title="Dieta",
        yaxis_title="Ganancia de peso (kg)",
    )
    st.plotly_chart(figura, width="stretch")
    mostrar_ejercicio(
        "Si el estadístico F es grande, ¿qué sugiere sobre las medias de las dietas?",
        [
            "Que la variación entre dietas es grande respecto al error",
            "Que todas las medias son necesariamente iguales",
            "Que no hubo aleatorización",
        ],
        "Que la variación entre dietas es grande respecto al error",
        "Un F grande aporta evidencia contra la igualdad de todas las medias; "
        "la conclusión formal se contrasta con el p-valor.",
    )


def pagina_dbca():
    st.title("🧱 DBCA — Diseño en Bloques Completamente al Azar")
    st.write(
        "El DBCA se usa cuando las unidades experimentales pueden agruparse en "
        "bloques relativamente homogéneos respecto de una fuente de variación "
        "conocida. Cada bloque recibe todos los tratamientos, asignados al azar "
        "dentro del bloque."
    )
    st.write(
        "El bloqueo no es el tratamiento de interés: su función es reducir el "
        "error experimental y mejorar la precisión de la comparación. El modelo "
        "aditivo habitual supone que no hay interacción tratamiento × bloque."
    )
    st.latex(r"Y_{ij} = \mu + \tau_i + \beta_j + \varepsilon_{ij}")
    st.caption(
        r"$\tau_i$: efecto del tratamiento; $\beta_j$: efecto del bloque; "
        r"$\varepsilon_{ij}$: error aleatorio."
    )
    fertilizantes = {
        "Fertilizante A": [18.2, 20.1, 19.0],
        "Fertilizante B": [21.4, 22.0, 20.8],
        "Fertilizante C": [24.1, 23.4, 25.0],
        "Fertilizante D": [20.0, 21.1, 19.7],
    }
    bloques = ["Bloque 1", "Bloque 2", "Bloque 3"]
    datos = pd.DataFrame(
        [
            {"Fertilizante": trat, "Bloque": bloques[j], "Rendimiento": valores[j]}
            for trat, valores in fertilizantes.items()
            for j in range(len(bloques))
        ]
    )
    st.markdown("### Ejemplo: cuatro fertilizantes en tres bloques")
    st.write(
        "Se comparan cuatro fertilizantes en tres bloques de campo; cada bloque "
        "contiene una parcela con cada fertilizante."
    )
    st.dataframe(datos, hide_index=True, width="stretch")
    modelo = smf.ols("Rendimiento ~ C(Fertilizante) + C(Bloque)", data=datos).fit()
    st.markdown("#### ANOVA con tratamiento y bloque")
    mostrar_anova(modelo)
    st.caption(
        "Con t tratamientos y b bloques: tratamiento t−1, bloque b−1, "
        "error (t−1)(b−1), total tb−1."
    )
    figura = px.line(
        datos,
        x="Fertilizante",
        y="Rendimiento",
        color="Bloque",
        markers=True,
        title="Respuesta de cada bloque a los tratamientos",
    )
    figura.update_layout(yaxis_title="Rendimiento (unidades)", xaxis_title="Fertilizante")
    st.plotly_chart(figura, width="stretch")
    mostrar_ejercicio(
        "¿Qué condición define al DBCA clásico?",
        [
            "Cada bloque contiene todos los tratamientos",
            "Cada tratamiento aparece en un solo bloque",
            "Los bloques se asignan como niveles de un segundo tratamiento",
        ],
        "Cada bloque contiene todos los tratamientos",
        "La completitud permite comparar los tratamientos dentro de cada bloque.",
    )


def pagina_dcl():
    st.title("🔲 DCL — Diseño en Cuadro Latino")
    st.write(
        "El cuadro latino controla dos fuentes de variación ajenas al tratamiento, "
        "representadas por filas y columnas. Cada tratamiento aparece exactamente "
        "una vez en cada fila y una vez en cada columna."
    )
    st.write(
        "Este doble control local es útil cuando existe, por ejemplo, un gradiente "
        "de fertilidad en dos direcciones. El diseño requiere igual número de "
        "tratamientos, filas y columnas, además de aleatorización compatible con "
        "la estructura del cuadro."
    )
    st.latex(r"Y_{ijk} = \mu + \tau_i + \beta_j + \gamma_k + \varepsilon_{ijk}")
    st.caption(
        r"$\beta_j$ y $\gamma_k$ representan los efectos de fila y columna; "
        r"$\tau_i$ es el efecto del tratamiento."
    )
    arreglo = np.array(
        [
            ["A", "B", "C", "D"],
            ["B", "C", "D", "A"],
            ["C", "D", "A", "B"],
            ["D", "A", "B", "C"],
        ]
    )
    efectos_trat = {"A": 0.0, "B": 2.0, "C": 4.0, "D": 1.0}
    efectos_fila = [0.0, 1.5, -0.5, 2.0]
    efectos_col = [0.5, -1.0, 1.0, 0.0]
    errores = [
        [0.2, -0.1, 0.0, -0.2],
        [-0.1, 0.3, -0.2, 0.1],
        [0.0, -0.2, 0.2, 0.1],
        [-0.2, 0.0, -0.1, 0.3],
    ]
    respuestas = []
    for i in range(4):
        for j in range(4):
            trat = arreglo[i, j]
            valor = (
                20 + efectos_trat[trat] + efectos_fila[i] + efectos_col[j]
                + errores[i][j]
            )
            respuestas.append(
                {"Fila": f"Fila {i + 1}", "Columna": f"Columna {j + 1}",
                 "Tratamiento": trat, "Respuesta": valor}
            )
    datos = pd.DataFrame(respuestas)
    st.markdown("### Ejemplo: cuadro latino 4 × 4")
    st.write("Se comparan los tratamientos A, B, C y D controlando dos gradientes de campo.")
    cuadrante = pd.DataFrame(
        arreglo,
        index=[f"Fila {i}" for i in range(1, 5)],
        columns=[f"Columna {i}" for i in range(1, 5)],
    )
    st.markdown("#### Arreglo de tratamientos")
    st.dataframe(cuadrante, width="stretch")
    modelo = smf.ols(
        "Respuesta ~ C(Tratamiento) + C(Fila) + C(Columna)", data=datos
    ).fit()
    st.markdown("#### ANOVA")
    mostrar_anova(modelo)
    mostrar_tabla_grados(
        [
            ("Tratamientos", 3),
            ("Filas", 3),
            ("Columnas", 3),
            ("Error", 6),
            ("Total", 15),
        ]
    )
    figura = px.imshow(
        datos.pivot(index="Fila", columns="Columna", values="Respuesta"),
        text_auto=".1f",
        color_continuous_scale="YlGn",
        aspect="auto",
        title="Respuesta observada según fila y columna",
    )
    st.plotly_chart(figura, width="stretch")
    mostrar_ejercicio(
        "En un cuadro latino de orden 4, ¿cuántas veces aparece cada tratamiento?",
        ["Una vez por fila y una vez por columna", "Cuatro veces en cada fila", "Solo una vez en todo el ensayo"],
        "Una vez por fila y una vez por columna",
        "Esa restricción permite ajustar simultáneamente los dos controles locales.",
    )


def anova_parcelas_divididas(datos):
    """Calcula la ANOVA de parcelas divididas con sus dos errores apropiados."""
    r = datos["Bloque"].nunique()
    a = datos["Riego"].nunique()
    b = datos["Variedad"].nunique()
    gran_media = datos["Rendimiento"].mean()
    media_bloque = datos.groupby("Bloque")["Rendimiento"].mean()
    media_riego = datos.groupby("Riego")["Rendimiento"].mean()
    media_variedad = datos.groupby("Variedad")["Rendimiento"].mean()
    media_bloque_riego = datos.groupby(["Bloque", "Riego"])["Rendimiento"].mean()
    media_riego_variedad = datos.groupby(["Riego", "Variedad"])["Rendimiento"].mean()

    ss_bloque = a * b * ((media_bloque - gran_media) ** 2).sum()
    ss_riego = r * b * ((media_riego - gran_media) ** 2).sum()
    ss_error_a = b * sum(
        (
            media_bloque_riego.loc[(bloque, riego)]
            - media_bloque.loc[bloque]
            - media_riego.loc[riego]
            + gran_media
        ) ** 2
        for bloque in media_bloque.index
        for riego in media_riego.index
    )
    ss_variedad = r * a * ((media_variedad - gran_media) ** 2).sum()
    ss_interaccion = r * sum(
        (
            media_riego_variedad.loc[(riego, variedad)]
            - media_riego.loc[riego]
            - media_variedad.loc[variedad]
            + gran_media
        ) ** 2
        for riego in media_riego.index
        for variedad in media_variedad.index
    )
    ss_total = ((datos["Rendimiento"] - gran_media) ** 2).sum()
    ss_error_b = ss_total - ss_bloque - ss_riego - ss_error_a - ss_variedad - ss_interaccion
    gl_error_a = (r - 1) * (a - 1)
    gl_error_b = a * (r - 1) * (b - 1)
    componentes = [
        ("Bloques", ss_bloque, r - 1, "—"),
        ("Riego (parcela principal)", ss_riego, a - 1, "Error(a)"),
        ("Error(a)", ss_error_a, gl_error_a, "—"),
        ("Variedad (subparcela)", ss_variedad, b - 1, "Error(b)"),
        ("Riego × variedad", ss_interaccion, (a - 1) * (b - 1), "Error(b)"),
        ("Error(b)", ss_error_b, gl_error_b, "—"),
        ("Total", ss_total, r * a * b - 1, "—"),
    ]
    salida = []
    ms_por_fuente = {
        fuente: ss / gl for fuente, ss, gl, _ in componentes if fuente != "Total"
    }
    for fuente, ss, gl, denominador in componentes:
        cm = ss / gl if gl else np.nan
        if fuente in ("Riego (parcela principal)",):
            f = cm / ms_por_fuente["Error(a)"]
            p = stats.f.sf(f, gl, gl_error_a)
        elif fuente in ("Variedad (subparcela)", "Riego × variedad"):
            f = cm / ms_por_fuente["Error(b)"]
            p = stats.f.sf(f, gl, gl_error_b)
        else:
            f, p = np.nan, np.nan
        salida.append(
            {
                "Fuente de variación": fuente,
                "Suma de cuadrados": ss,
                "Grados de libertad": gl,
                "Cuadrado medio": cm,
                "F": f,
                "p-valor": p,
                "Denominador de F": denominador,
            }
        )
    return pd.DataFrame(salida)


def pagina_dpd():
    st.title("🌱 DPD — Diseño en Parcelas Divididas")
    st.write(
        "En parcelas divididas, un factor difícil o costoso de cambiar se asigna "
        "a parcelas principales; dentro de ellas se aleatorizan los niveles de un "
        "segundo factor en subparcelas. La jerarquía refleja restricciones reales "
        "de operación, como el uso de sistemas de riego."
    )
    st.write(
        "La estructura implica dos precisiones: el factor de parcela principal se "
        "contrasta con el error(a), mientras que el factor de subparcela y su "
        "interacción se contrastan con el error(b). No se deben intercambiar esos "
        "denominadores en las pruebas F."
    )
    st.latex(
        r"Y_{ijk} = \mu + \rho_i + \alpha_j + e_{ij}^{(a)} "
        r"+ \beta_k + (\alpha\beta)_{jk} + e_{ijk}^{(b)}"
    )
    st.caption(
        r"$\alpha_j$: factor de parcela principal (riego); $\beta_k$: factor de "
        r"subparcela (variedad); $\rho_i$: bloque."
    )
    medias = {
        ("Riego alto", "Variedad A"): 32.0,
        ("Riego alto", "Variedad B"): 35.0,
        ("Riego alto", "Variedad C"): 33.0,
        ("Riego bajo", "Variedad A"): 26.0,
        ("Riego bajo", "Variedad B"): 29.0,
        ("Riego bajo", "Variedad C"): 28.0,
    }
    ajustes = [
        [0.0, 0.7, -0.4, 0.3, -0.6, 0.2],
        [0.5, -0.2, 0.6, -0.5, 0.4, -0.1],
        [-0.3, 0.1, -0.2, 0.6, -0.1, 0.5],
    ]
    combinaciones = list(medias)
    filas = []
    for i, bloque in enumerate(["Bloque 1", "Bloque 2", "Bloque 3"]):
        for j, (riego, variedad) in enumerate(combinaciones):
            filas.append(
                {
                    "Bloque": bloque,
                    "Riego": riego,
                    "Variedad": variedad,
                    "Rendimiento": medias[(riego, variedad)] + ajustes[i][j],
                }
            )
    datos = pd.DataFrame(filas)
    st.markdown("### Ejemplo agrícola: riego × variedad")
    st.write(
        "En tres bloques, se asigna riego alto o bajo a parcelas principales y "
        "tres variedades a las subparcelas."
    )
    st.dataframe(datos, hide_index=True, width="stretch")
    tabla = anova_parcelas_divididas(datos)
    st.markdown("#### ANOVA con error(a) y error(b)")
    st.dataframe(tabla.round(4), hide_index=True, width="stretch")
    figura = px.line(
        datos.groupby(["Riego", "Variedad"], as_index=False)["Rendimiento"].mean(),
        x="Variedad",
        y="Rendimiento",
        color="Riego",
        markers=True,
        title="Rendimiento medio por combinación de riego y variedad",
    )
    st.plotly_chart(figura, width="stretch")
    mostrar_ejercicio(
        "¿Con qué término de error se prueba el efecto de riego?",
        ["Error(a)", "Error(b)", "Error total sin particionar"],
        "Error(a)",
        "Riego se asigna a parcelas principales, por eso su variación experimental "
        "se estima con error(a).",
    )


def pagina_factoriales():
    st.title("⚗️ Diseños Factoriales")
    st.write(
        "Un diseño factorial estudia simultáneamente dos o más factores y observa "
        "todas sus combinaciones de niveles. Esto permite estimar los efectos "
        "principales y determinar si el efecto de un factor depende del nivel de otro."
    )
    st.write(
        "La interacción es clave para interpretar los resultados: si es importante, "
        "los efectos principales aislados pueden resumir mal el comportamiento. "
        "La gráfica de interacción ayuda a detectar visualmente líneas no paralelas."
    )
    st.latex(
        r"Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}"
    )
    st.caption(
        r"$\alpha_i$ y $\beta_j$ son efectos principales; $(\alpha\beta)_{ij}$ "
        r"es la interacción."
    )
    datos = []
    medias = {
        ("Baja", "Cat. A"): 68,
        ("Baja", "Cat. B"): 74,
        ("Baja", "Cat. C"): 72,
        ("Alta", "Cat. A"): 76,
        ("Alta", "Cat. B"): 80,
        ("Alta", "Cat. C"): 91,
    }
    desviaciones = [-1.2, 0.4, 1.1, -0.6]
    for temperatura in ["Baja", "Alta"]:
        for catalizador in ["Cat. A", "Cat. B", "Cat. C"]:
            for replica, desvio in enumerate(desviaciones, start=1):
                datos.append(
                    {
                        "Temperatura": temperatura,
                        "Catalizador": catalizador,
                        "Réplica": replica,
                        "Rendimiento": medias[(temperatura, catalizador)] + desvio,
                    }
                )
    datos = pd.DataFrame(datos)
    st.markdown("### Ejemplo: temperatura × catalizador")
    st.write(
        "Se evalúan dos temperaturas y tres catalizadores, con cuatro réplicas "
        "por combinación, para estudiar el rendimiento de una reacción."
    )
    st.dataframe(datos, hide_index=True, width="stretch")
    modelo = smf.ols(
        "Rendimiento ~ C(Temperatura) * C(Catalizador)", data=datos
    ).fit()
    st.markdown("#### ANOVA factorial")
    mostrar_anova(modelo)
    st.caption(
        "Para un factorial a × b con n réplicas: A tiene a−1 gl, B b−1, "
        "A×B (a−1)(b−1), error ab(n−1)."
    )
    figura = px.line(
        datos.groupby(["Temperatura", "Catalizador"], as_index=False)["Rendimiento"].mean(),
        x="Catalizador",
        y="Rendimiento",
        color="Temperatura",
        markers=True,
        title="Gráfica de interacción: temperatura × catalizador",
    )
    st.plotly_chart(figura, width="stretch")
    mostrar_ejercicio(
        "¿Qué patrón visual puede sugerir interacción entre dos factores?",
        [
            "Líneas aproximadamente paralelas",
            "Líneas que no son paralelas o se cruzan",
            "Una única media para todos los grupos",
        ],
        "Líneas que no son paralelas o se cruzan",
        "La falta de paralelismo sugiere que la respuesta a un factor cambia según "
        "el nivel del otro; la ANOVA evalúa formalmente esa interacción.",
    )


def crear_datos_ccd():
    """Construye un diseño central compuesto replicado en el centro."""
    alfa = np.sqrt(2)
    puntos = [
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-alfa, 0), (alfa, 0), (0, -alfa), (0, alfa),
        (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
    ]
    observaciones = []
    for x1, x2 in puntos:
        respuesta = (
            80 + 5 * x1 + 3 * x2 - 8 * x1**2 - 5 * x2**2
            + 4 * x1 * x2 + 0.7 * np.sin(3 * x1 + 2 * x2)
        )
        observaciones.append({"X1": x1, "X2": x2, "Rendimiento": respuesta})
    return pd.DataFrame(observaciones)


def pagina_msr():
    st.title("🏔️ MSR — Metodología de Superficies de Respuesta")
    st.write(
        "La MSR combina herramientas de diseño y modelado para entender cómo "
        "cambian las respuestas ante factores cuantitativos y para localizar "
        "condiciones de operación convenientes. El diseño debe cubrir la región "
        "experimental con puntos informativos."
    )
    st.write(
        "Un modelo de primer orden aproxima tendencias lineales y suele usarse "
        "para explorar una región. Si aparece curvatura, un modelo de segundo "
        "orden añade términos cuadrados e interacciones; un diseño central "
        "compuesto aporta puntos factoriales, axiales y centrales para estimarlo."
    )
    st.latex(
        r"Y = \beta_0 + \sum_i \beta_i X_i + \sum_i \beta_{ii} X_i^2 "
        r"+ \sum_{i<j} \beta_{ij} X_i X_j + \varepsilon"
    )
    st.latex(r"Y = \beta_0 + \sum_i \beta_i X_i + \varepsilon \qquad \text{(primer orden)}")
    st.markdown("### Ejemplo: diseño central compuesto con dos factores")
    st.write(
        "Se estudia el rendimiento en función de temperatura (X₁) y concentración "
        "(X₂), codificadas en unidades centradas y escaladas. El conjunto ilustrativo "
        "incluye puntos factoriales, axiales y cinco réplicas centrales."
    )
    datos = crear_datos_ccd()
    st.dataframe(datos.round(3), hide_index=True, width="stretch")
    modelo = smf.ols(
        "Rendimiento ~ X1 + X2 + I(X1**2) + I(X2**2) + X1:X2", data=datos
    ).fit()
    st.markdown("#### ANOVA del modelo cuadrático")
    mostrar_anova(modelo)
    st.caption(
        "En un CCD de dos factores se combinan puntos factoriales, axiales y "
        "réplicas centrales; estas últimas ayudan a estimar el error puro."
    )

    eje = np.linspace(-1.5, 1.5, 60)
    x1_grid, x2_grid = np.meshgrid(eje, eje)
    rejilla = pd.DataFrame({"X1": x1_grid.ravel(), "X2": x2_grid.ravel()})
    z = modelo.predict(rejilla).to_numpy().reshape(x1_grid.shape)
    izquierda, derecha = st.columns(2)
    with izquierda:
        contorno = go.Figure(
            go.Contour(
                x=eje,
                y=eje,
                z=z,
                colorscale="Viridis",
                colorbar={"title": "Rendimiento"},
                contours={"showlabels": True},
            )
        )
        contorno.add_trace(
            go.Scatter(
                x=datos["X1"], y=datos["X2"], mode="markers",
                marker={"color": "white", "line": {"color": "black", "width": 1}},
                name="Puntos del diseño",
            )
        )
        contorno.update_layout(title="Contornos de respuesta", xaxis_title="X₁", yaxis_title="X₂")
        st.plotly_chart(contorno, width="stretch")
    with derecha:
        superficie = go.Figure(
            go.Surface(x=x1_grid, y=x2_grid, z=z, colorscale="Viridis")
        )
        superficie.update_layout(
            title="Superficie de respuesta ajustada",
            scene={"xaxis_title": "X₁", "yaxis_title": "X₂", "zaxis_title": "Rendimiento"},
        )
        st.plotly_chart(superficie, width="stretch")
    mostrar_ejercicio(
        "¿Qué añade el modelo de segundo orden frente al modelo de primer orden?",
        [
            "Términos cuadráticos que permiten representar curvatura",
            "La eliminación del error experimental",
            "Únicamente más niveles para factores categóricos",
        ],
        "Términos cuadráticos que permiten representar curvatura",
        "Los términos Xᵢ² permiten describir curvatura y apoyar la búsqueda de "
        "un óptimo dentro de la región estudiada.",
    )


PAGINAS = {
    "Inicio": pagina_inicio,
    "DCA": pagina_dca,
    "DBCA": pagina_dbca,
    "DCL": pagina_dcl,
    "DPD": pagina_dpd,
    "Diseños Factoriales": pagina_factoriales,
    "MSR": pagina_msr,
}


def main():
    """Configura la navegación y muestra la página seleccionada."""
    st.sidebar.title("📚 Navegación")
    seleccion = st.sidebar.radio(
        "Elige un diseño experimental",
        list(DISENOS),
        format_func=lambda nombre: f"{DISENOS[nombre]['icono']} {nombre}",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Material de apoyo para el curso de Diseño Experimental.")
    PAGINAS[seleccion]()
    st.markdown("---")
    st.warning("⚠️ Recuerda detener tu Codespace al terminar para no consumir horas")


if __name__ == "__main__":
    main()
