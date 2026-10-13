# Aula interactiva de Diseño Experimental

Aplicación didáctica en Streamlit para explorar diseños experimentales mediante
modelos estadísticos, tablas ANOVA, ejemplos reproducibles, gráficas y preguntas
de práctica.

## Diseños incluidos

- Diseño Completamente al Azar (DCA)
- Diseño en Bloques Completamente al Azar (DBCA)
- Diseño en Cuadro Latino (DCL)
- Diseño en Parcelas Divididas (DPD)
- Diseños Factoriales
- Metodología de Superficies de Respuesta (MSR)

## Requisitos

Se recomienda Python 3.9 o posterior y `pip`.

## Instalación y ejecución

Desde la carpeta del proyecto, instala las dependencias:

```bash
python -m pip install -r requirements.txt
```

Inicia la aplicación con el comando:

```bash
streamlit run app.py
```

Streamlit mostrará una dirección local en la terminal (habitualmente
`http://localhost:8501`) para abrir la aplicación en el navegador.

## Detener la aplicación y el Codespace

- **Aplicación:** vuelve a la terminal donde ejecutaste Streamlit y presiona
  `Ctrl+C`.
- **Codespace desde GitHub:** abre el menú **Code → Codespaces**, localiza el
  Codespace activo y selecciona **Stop codespace**.

Detener Streamlit no detiene el Codespace; realiza ambas acciones cuando hayas
terminado.
