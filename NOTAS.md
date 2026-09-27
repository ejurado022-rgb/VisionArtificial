# Notas del proyecto

Estas son mis notas de trabajo sobre VisionArtificial: las decisiones que
tomé y el porqué, para no perder el hilo si vuelvo a este código más
adelante.

Documentación completa:

- `docs/SPEC.md` — especificación funcional, alcance y resultados.
- `docs/ARCHITECTURE.md` — arquitectura técnica (los 5 módulos del sistema).
- `docs/PLAN.md` — plan de implementación por fases.

## Decisiones que tomé

**Las 3 frutas son manzana, banano y naranja.** El documento de la propuesta
no las fijaba de antemano; elegí estas tres porque son fáciles
de conseguir y porque coinciden con las clases de un dataset público de
Roboflow Universe que pude aprovechar en vez de depender solo de fotos
propias (el porqué completo está en `docs/SPEC.md` §2).

**Dividí el trabajo en dos etapas.** Primero dejé funcionando un sistema
completo con un dataset más chico pero real: detección, conteo, métricas
y un CLI de demo de punta a punta. El diseño experimental más ambicioso
—27 combinaciones de iluminación, agrupamiento y ángulo, 810 imágenes,
análisis estadístico con ANOVA— lo dejé para una segunda etapa, porque
implica salir a capturar fotos en condiciones controladas y no se resuelve
en una sesión de trabajo.

**El dataset y los pesos entrenados no viven en este repositorio.** Pesan
demasiado y se regeneran sin problema: el dataset se gestiona en Roboflow
y los pesos salen de correr `scripts/train.py` sobre él.

**`counter.py` y `metrics.py` son lógica pura, sin entrada/salida.** Los
puedo probar con datos inventados sin necesitar un modelo entrenado de
verdad. Le presto especial atención a `counter.py`: un error ahí produce
un conteo de frutas incorrecto sin que se note a simple vista, así que
prefiero tenerlo bien cubierto de pruebas.

## Cómo trabajo en este repo

Python 3.10+. El paquete base no trae dependencias pesadas
(`pip install -e .[dev]` alcanza para correr los tests); `ultralytics`,
`opencv-python`, `pandas` y `matplotlib` viven en el extra `ml`
(`pip install -e .[dev,ml]`) y solo hacen falta para entrenar o correr el
modelo. Sigo el orden de fases de `docs/PLAN.md`.
