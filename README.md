# VRC / HRV RRi Analyzer Pro v8.2

Mejora principal:
- Nueva representación visual HRV por fases en 6 paneles:
 1) RMSSD, SDNN, pNN50
 2) VLF, LF, HF, TOTAL
 3) SD1, SD2
 4) DFA α1, DFA α2, D2, ApEn, SampEn
 5) Recurrence Plot
 6) MSE 1-20
- Columnas verticales por fase.
- Líneas de tendencia suavizadas superpuestas.
- Incluido en pestaña HRV y Dashboard.

Subir a GitHub:
- app.py
- requirements.txt
- README.md


## v7.4
- Leyendas del resumen HRV fuera de los gráficos.
- Paneles izquierdos: leyenda en el margen entre columnas.
- Paneles derechos: leyenda en el margen derecho.
- Colores fijos globales para todos los parámetros.
- MSE con leyenda agrupada:
 - MSE corto plazo (1-5)
 - MSE medio plazo (6-12)
 - MSE largo plazo (13-20)


## v7.5
- Corrección de líneas de tendencia suavizadas.
- Con 3 fases se usa ajuste cuadrático para que la curva no salga recta.
- Con 4 o más fases se usa CubicSpline natural.


## v7.6
- Corrige el error de Plotly: `legend2` / `legend3` no soportado en algunas versiones.
- Sustituye leyendas múltiples por leyendas manuales con anotaciones.
- Mantiene leyendas fuera de los gráficos.
- Mantiene líneas suavizadas.


## v7.7
- Corrige el error de Plotly en Dashboard cuando se seleccionan muchos parámetros.
- `horizontal_spacing` ahora es dinámico y seguro.
- Dominios Amplitud/Vagal/Complejidad/Recurrencia con líneas suavizadas reales.
- Mantiene leyendas manuales fuera de los gráficos y colores fijos.


## v7.8
- Eliminado el texto "Kubios Mode" del título y configuración de la app.


## v7.9
- Añadida clasificación del tipo de grafo HVG:
  - Libre de escala / jerárquico
  - Small-world funcional
  - Lineal / cadena
  - Regular / homogéneo
  - Complejo mixto
- Añadidas puntuaciones orientativas:
  - score libre de escala
  - score small-world
  - score cadena/lineal
- Ampliada la sección Métricas HVG / grafos con explicación clínica de cada métrica.


## v8.0
- Corrige el TypeError en Dominios cuando hay columnas de texto como `HVG_graph_type`.
- `domain_values` ahora sólo usa variables numéricas.
- Mantiene clasificación del tipo de grafo HVG y tabla explicativa ampliada.


## v8.1
- Corrige el error del informe automático cuando aparecen métricas textuales como `HVG_graph_type`.
- `_interpret_metric`, `_fmt_num` y `_arrow_change` ahora aceptan valores no numéricos.
- Mantiene la clasificación del tipo de grafo HVG en la tabla explicativa.

## v8.2
- Corrige el Dashboard comparativo para que muestre columnas verticales visibles.
- La línea suavizada se superpone sobre las barras.
- En una fase con varios registros, el eje X usa posiciones numéricas con etiquetas de registro.
