# cv_experience_ocr_bert

Sistema de punta a punta que lee **imágenes de CVs** con **PaddleOCR** y predice los **años de experiencia laboral (0–20)** con un transformer ajustado como regresor (**ModernBERT**), comparado contra tres líneas base: mediana, TF-IDF + Ridge y embeddings congelados + Ridge.

> 🚧 En construcción. Cada sección se completa en su propia rama y Pull Request.

## Contenido

1. Proyecto y objetivo
2. Datos y muestreo
3. Configuración del OCR
4. Limpieza y enmascarado
5. Split
6. Líneas base
7. Configuración de BERT
8. Métricas finales
9. Preguntas analíticas (Q1–Q4)
10. Limitaciones
11. Reproducción y app

## Decisiones técnicas (resumen)

| Decisión | Elección | Justificación breve |
|---|---|---|
| Muestra | 4,000 CVs sin foto (800 por ocupación), semilla 42 | Potencia estadística; el cómputo pesado corre en una GPU de Colab |
| Tope del objetivo | 20 años (se revisa con la regla del 15%) | — |
| OCR | PaddleOCR 3.7, `PP-OCRv6_medium` (det + rec), orientación, desenrollado y orientación de línea apagados | En 50 CVs sobre una T4: CER 0.00% y WER 0.03%, frente a 0.35%/5.4% (v5 mobile) y 0.70%/8.6% (v5 server); 0.54 s/CV. Imágenes sintéticas planas y derechas |
| Orden de lectura | Detección de columnas por agrupación en x | Mantiene cada fecha junto a su puesto |
| Confianza baja | No se descartan líneas (variante en Q3) | Las líneas con fechas tienden a tener menor confianza |
| Enmascarado | Regex conservadora (experiencia total, edad, fecha de nacimiento) | Evita fuga de información sin borrar fechas de trabajo |
| TF-IDF | n-gramas de palabras 1–2, dígitos conservados | Línea base fuerte |
| Embeddings | `bge-small-en-v1.5`, trozos de 512 tokens y promedio | — |
| Transformer | `answerdotai/ModernBERT-base`, contexto largo | El CV completo cabe en el contexto; no hace falta truncar |
| Pérdida | MSE (principal); Huber como experimento de Q3 | Obligatoria; etiquetas sintéticas exactas |

## Entorno

- **Laptop (CPU):** Python 3.12, entorno virtual `.venv`. Limpieza, split, líneas base, evaluación y app.
- **Google Colab (GPU T4):** OCR masivo y fine-tuning de ModernBERT. Colab clona este repo, corre los scripts de `src/` y guarda las salidas en Google Drive. Usa `requirements-colab.txt`, que fija solo las librerías del proyecto y respeta el stack científico que trae Colab.

```bash
py -V:3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # y pega tu HF_TOKEN
```

> **Nota (Windows):** en cualquier script que use PaddleOCR y PyTorch juntos, hay que importar `torch` **antes** que `paddle`/`paddleocr`. En el orden inverso, sus DLLs chocan (`WinError 127`).

Las imágenes, la salida del OCR, el texto procesado y los pesos de los modelos **no se versionan** (ver `.gitignore`). Se regeneran con los scripts de `src/`.
